"""Bounded YOLO11n training with target-preserving augmentation and real callback logs."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def configure_runtime(root):
    """Isolate Ultralytics/Matplotlib directories and enforce offline deterministic runtime."""
    root = Path(root)
    config = root / "ultralytics-config"
    (config / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (root / "matplotlib-config").mkdir(exist_ok=True)
    os.environ.update(
        YOLO_CONFIG_DIR=str(config),
        MPLCONFIGDIR=str(root / "matplotlib-config"),
        YOLO_OFFLINE="true",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    from ultralytics.utils import SETTINGS

    SETTINGS.update({k: False for k in ["sync", "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune", "wandb"]})
    SETTINGS.update({
        "datasets_dir": str(root / "datasets"),
        "weights_dir": str(root / "downloads"),
        "runs_dir": str(root / "runs"),
    })


def completed_epochs(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return max(
            (int(float({k.strip(): v for k, v in row.items()}["epoch"])) for row in csv.DictReader(stream)),
            default=0,
        )


def trainer_class():
    from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
    from ultralytics.models.yolo.detect import DetectionTrainer

    class TargetPreservingTrainer(DetectionTrainer):
        def optimizer_step(self):
            import torch

            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                raise FloatingPointError("Missing/nonfinite gradients before optimizer step")
            self.checked_optimizer_gradients = True
            return super().optimizer_step()

        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                ds.transforms = Compose([
                    LetterBox((self.args.imgsz, self.args.imgsz), scaleup=False),
                    RandomHSV(self.args.hsv_h, self.args.hsv_s, self.args.hsv_v),
                    RandomFlip(self.args.flipud, "vertical"),
                    RandomFlip(self.args.fliplr, "horizontal"),
                    Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0),
                ])
            return ds

    return TargetPreservingTrainer


def run(plan_path, smoke=False):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    root = Path(plan["task_root"])
    configure_runtime(root)

    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    from pcb_vlm.data.yolo_v3 import read, write
    from pcb_vlm.provenance import code_identity, sha256, source_identity

    for file, digest in plan["input_sha256"].items():
        if sha256(file) != digest:
            raise ValueError(f"Frozen input changed: {file}")
    if source_identity() != plan["source_tree_sha256"]:
        raise ValueError("Frozen source changed")

    folder = root / ("smoke" if smoke else "training")
    if folder.exists():
        raise FileExistsError(f"Output directory already exists: {folder}")

    initial = {}
    model = YOLO(plan["initial_weights"])
    events = []
    start = time.monotonic()
    torch.set_num_threads(2)

    def on_train_start(trainer):
        expected_nc = len(trainer.data["names"])
        if trainer.model.nc != expected_nc:
            raise ValueError(f"YOLO class count mismatch: {trainer.model.nc} != {expected_nc}")
        initial["parameter"] = next(p for p in trainer.model.parameters() if p.requires_grad).detach().cpu().clone()

        # Prove actual transformed target count is preserved, including negatives/slivers.
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_rng)
        expected = {str(Path(r["image_path"]).resolve()): r for r in read(root / "yolo_data/provenance.json")["records"]}
        counts = []
        data = trainer.train_loader.dataset
        for i in range(len(data)):
            record = expected[str(Path(data.im_files[i]).resolve())]
            before = len(data.labels[i]["cls"])
            after = len(data[i]["cls"])
            if not (before == after == len(record["objects"])):
                raise ValueError(f"Transformed target count mismatch for {record['id']}: {before}/{after}/{len(record['objects'])}")
            counts.append([record["id"], before, after])
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3])

        write(folder / "runtime_model.json", {
            "version": ultralytics.__version__,
            "git_commit": code_identity(),
            "source_sha256": source_identity(),
            "input_plan_sha256": sha256(plan_path),
            "model_params": sum(p.numel() for p in trainer.model.parameters()),
            "class_names": trainer.data["names"],
            "sample_transform_count_checks": counts,
            "transform_pipeline": str(trainer.train_loader.dataset.transforms),
            "args": vars(trainer.args),
            "augmentation_policy": (
                "No Mosaic/affine/scale/translation/crop/albumentations. "
                "LetterBox + HSV + horizontal/vertical flip preserve all labels."
            ),
        })

    def on_train_batch_end(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Nonfinite training loss")
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Nonfinite gradients")
        if smoke and len(events) >= 2:
            trainer.stop = True
        peak_alloc = round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0
        peak_res = round(torch.cuda.max_memory_reserved() / 2**20, 2) if torch.cuda.is_available() else 0.0
        record = {
            "epoch": trainer.epoch + 1,
            "batch": len(events) % len(trainer.train_loader),
            "loss": float(trainer.loss.detach().sum()),
            "gradients_present": bool(grads),
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_res,
            "wall_s": round(time.monotonic() - start, 3),
        }
        events.append(record)
        with (folder / "batch-events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def epoch_end(trainer):
        if trainer.epoch + 1 > completed_epochs(folder / "results.csv"):
            return  # final_eval callback is not another trained epoch
        peak_alloc = round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0
        record = {
            "epoch": trainer.epoch + 1,
            "metrics": trainer.metrics,
            "fitness": float(trainer.fitness),
            "best_fitness": float(trainer.best_fitness),
            "peak_allocated_mib": peak_alloc,
            "wall_s": round(time.monotonic() - start, 3),
        }
        with (folder / "epoch-events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print("EPOCH_AUDIT", json.dumps(record), flush=True)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", epoch_end)

    kwargs = dict(plan["training"])
    kwargs.update(
        data=plan["smoke_dataset"] if smoke else plan["dataset_yaml"],
        project=str(root),
        name=folder.name,
        exist_ok=False,
    )
    if smoke:
        kwargs.update(epochs=1, workers=0, plots=False, patience=0, val=True)
    model.train(trainer=trainer_class(), **kwargs)

    trainer = model.trainer
    best = Path(trainer.best)
    last = Path(trainer.last)
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("Model checkpoint was not saved")

    reloaded = YOLO(str(best))
    expected_nc = len(trainer.data["names"])
    if reloaded.task != "detect" or len(reloaded.names) != expected_nc:
        raise ValueError(f"Reloaded checkpoint task/classes mismatch: {reloaded.task}, {len(reloaded.names)}")

    peak_alloc = round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0
    write(folder / "run_receipt.json", {
        "completed": True,
        "smoke": smoke,
        "initial_weights_sha256": sha256(plan["initial_weights"]),
        "best_path": str(best),
        "best_sha256": sha256(best),
        "last_path": str(last),
        "last_sha256": sha256(last),
        "epochs_completed": completed_epochs(folder / "results.csv"),
        "updates": trainer.ema.updates if trainer.ema else None,
        "best_criterion": "Ultralytics 8.4.68 Metric.fitness: mAP50:95 on internal Val tiles, weights [0,0,0,1]. Not full-image GT result.",
        "best_reloaded": True,
        "trainable_parameter_changed": not torch.equal(
            initial["parameter"],
            next(p for p in trainer.model.parameters() if p.requires_grad).detach().cpu(),
        ),
        "peak_allocated_mib": peak_alloc,
        "wall_s": round(time.monotonic() - start, 3),
        "batch_events": len(events),
        "gradients_checked_before_optimizer_step": getattr(trainer, "checked_optimizer_gradients", False),
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.plan, a.smoke)
