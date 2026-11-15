"""CPU HTTP/API & scope-isolation tests for scripts/fics_showcase.py."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import fics_showcase  # noqa: E402


def _make_mock_showcase_index(tmp_dir: pathlib.Path) -> pathlib.Path:
    def _build_ds(ds_key: str, board_id: str, shared_img_id: int, pred_box: list[float]) -> dict:
        d = tmp_dir / ds_key
        d.mkdir(parents=True)
        roi_p = d / f"{board_id}_roi.png"
        Image.new("RGB", (200, 120), color=(40, 80, 120)).save(roi_p)

        coco = {
            "images": [{"id": shared_img_id, "file_name": f"{board_id}.png", "width": 200, "height": 120, "board_id": board_id}],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [{"id": 1, "image_id": shared_img_id, "category_id": 6, "bbox": [10, 10, 40, 40]}],
        }
        coco_p = d / "gt_coco.json"
        coco_p.write_text(json.dumps(coco), encoding="utf-8")

        plan = {"models": {"qwen3": {}}, "boards": {board_id: {"image_id": shared_img_id}}}
        plan_p = d / "frozen_plan.json"
        plan_p.write_text(json.dumps(plan), encoding="utf-8")

        raw_dir = d / "raw"
        raw_dir.mkdir()
        raw_entry = {
            "call_id": f"call_{board_id}_W",
            "board_id": board_id,
            "mode": "W",
            "orig_input_width": 200,
            "orig_input_height": 120,
            "roi_offset_x": 0,
            "roi_offset_y": 0,
            "image_path": str(roi_p),
            "expected_sha256": hashlib.sha256(roi_p.read_bytes()).hexdigest(),
            "generated_tokens": 12,
            "max_new_tokens": 512,
            "generated_token_ids": [1, 2, 151645],
            "gpu_generate_ms": 100.0,
            "cuda_max_reserved_mib": 1000.0,
            "raw_output": '[{"bbox_2d":[50,83,250,417],"label":"ic"}]<|im_end|>',
        }
        raw_files = {}
        for mk in ("qwen2", "deepseek", "qwen3"):
            rp = raw_dir / f"{mk}.jsonl"
            entry = dict(raw_entry)
            if mk != "qwen3":
                entry["raw_output"] = "non json output"
            rp.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            raw_files[mk] = {"path": str(rp), "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}

        model_block = {}
        for mk in ("qwen2", "deepseek", "qwen3"):
            preds = [{"label": "ic", "bbox_xyxy": pred_box}] if (mk == "qwen3" and pred_box) else []
            tp = 1 if preds else 0
            fn = 1 - tp
            pb = {
                board_id: {
                    mode: {
                        "post_nms_predictions": len(preds) if mode == "W" else 0,
                        "pred_record": {"image_id": shared_img_id, "objects": preds if mode == "W" else []},
                        "evaluate_detections_iou50": {
                            "tp": tp if mode == "W" else 0,
                            "fp": 0,
                            "fn": fn if mode == "W" else 1,
                            "total_gt": 1,
                            "total_predictions": len(preds) if mode == "W" else 0,
                            "micro": {"precision": float(tp if mode == "W" else 0), "recall": float(tp if mode == "W" else 0), "f1": float(tp if mode == "W" else 0)},
                            "per_class": {},
                        },
                    }
                    for mode in ("W", "T")
                }
            }
            model_block[mk] = {
                "termination_v2": {
                    "total_calls_observed": 1,
                    "completed_ok_calls": 1,
                    "normal_eos": 1,
                    "length_limited_without_eos": 0,
                    "syntax_or_format_incomplete_after_eos": 0 if mk == "qwen3" else 1,
                    "strict_schema_ok": 1 if mk == "qwen3" else 0,
                    "explicit_empty": 0,
                },
                "per_board": pb,
                "combined": {mode: pb[board_id][mode] for mode in ("W", "T")},
            }
        rep_p = d / "replay.json"
        rep_p.write_text(json.dumps({"models": model_block}), encoding="utf-8")
        return {
            "label": ds_key,
            "split_role": ds_key,
            "artifacts": {
                "replay_report": {"path": str(rep_p), "sha256": hashlib.sha256(rep_p.read_bytes()).hexdigest()},
                "gt_coco": {"path": str(coco_p), "sha256": hashlib.sha256(coco_p.read_bytes()).hexdigest()},
                "frozen_plan": {"path": str(plan_p), "sha256": hashlib.sha256(plan_p.read_bytes()).hexdigest()},
            },
            "raw_jsonl": raw_files,
        }

    idx = {
        "schema_version": "1.0",
        "datasets": {
            # Deliberately reuse identical coco image_id=18 across two scopes to verify scope isolation!
            "scope_a": _build_ds("scope_a", "board_a", 18, [10.0, 10.0, 50.0, 50.0]),
            "scope_b": _build_ds("scope_b", "board_b", 18, [12.0, 12.0, 48.0, 48.0]),
        },
    }
    idx_p = tmp_dir / "results_index.json"
    idx_p.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return idx_p


class TestFicsShowcase(unittest.TestCase):
    def test_scope_isolation_missing_error_and_jpeg_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idx_p = _make_mock_showcase_index(pathlib.Path(tmp))
            srv = fics_showcase.create_server(index_path=idx_p, host="127.0.0.1", port=0)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base}/api/overview") as r:
                    ov = json.loads(r.read().decode("utf-8"))
                    self.assertIn("scope_a", ov["datasets"])
                    self.assertIn("scope_b", ov["datasets"])

                with urllib.request.urlopen(f"{base}/api/board?dataset=scope_a&board=board_a&model=qwen3&mode=W") as r:
                    ba = json.loads(r.read().decode("utf-8"))
                    self.assertEqual(ba["board_id"], "board_a")
                    self.assertEqual(ba["metrics"]["tp"], 1)

                # Must NOT cross-leak board_b into scope_a!
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base}/api/board?dataset=scope_a&board=board_b&model=qwen3&mode=W")
                self.assertEqual(ctx.exception.code, 404)

                # Image + crop JPEG route
                with urllib.request.urlopen(f"{base}/api/image?dataset=scope_b&board=board_b&crop=0,0,100,80") as r:
                    jpg = r.read()
                    self.assertTrue(jpg.startswith(b"\xff\xd8"))
            finally:
                srv.shutdown()
                srv.server_close()
                t.join(timeout=2.0)

    def test_counterexample_matching_parity_with_evaluate_py(self) -> None:
        import evaluate as repo_eval

        # Counterexample where global-max-IoU produces 2 TP while evaluate.py greedy pred-order produces 1 TP:
        # GT0=[0,0,100,100], GT1=[40,0,140,100]
        # P0=[18,0,118,100] overlaps GT0 (0.6949) > GT1 (0.6393) -> evaluate.py assigns P0->GT0
        # P1=[0,0,100,100] overlaps GT0 (1.0) and GT1 only at 0.4286 (<0.50) -> left unmatched (FP) => 1 TP, 1 FP, 1 FN
        gts = [
            {"id": 1, "label": "ic", "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]},
            {"id": 2, "label": "ic", "bbox_xyxy": [40.0, 0.0, 140.0, 100.0]},
        ]
        preds = [
            {"label": "ic", "bbox_xyxy": [18.0, 0.0, 118.0, 100.0]},
            {"label": "ic", "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]},
        ]
        gt_coco = {
            "images": [{"id": 1, "width": 200, "height": 200}],
            "categories": [{"id": 6, "name": "ic"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 6, "bbox": [0.0, 0.0, 100.0, 100.0]},
                {"id": 2, "image_id": 1, "category_id": 6, "bbox": [40.0, 0.0, 100.0, 100.0]},
            ],
        }
        ev = repo_eval.evaluate_detections(gt_coco, {"predictions": [{"image_id": 1, "status": "success", "objects": preds}]}, iou_threshold=0.50)
        gts_m, preds_m = fics_showcase._match_board_tp_fp_fn(gts, preds, iou_thresh=0.50)

        self.assertEqual(ev["tp"], 1)
        self.assertEqual(ev["fp"], 1)
        self.assertEqual(ev["fn"], 1)
        self.assertEqual(sum(1 for p in preds_m if p["match_status"] == "TP"), ev["tp"])
        self.assertEqual(sum(1 for p in preds_m if p["match_status"] == "FP"), ev["fp"])
        self.assertEqual(sum(1 for g in gts_m if g["match_status"] == "FN"), ev["fn"])
        self.assertEqual(preds_m[0]["matched_gt_id"], 1)
        self.assertIsNone(preds_m[1]["matched_gt_id"])

    def test_relative_index_paths_and_sha256_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            idx_p = _make_mock_showcase_index(tpath)
            idx_data = json.loads(idx_p.read_text(encoding="utf-8"))
            # Rewrite to relative paths against index_path.parent
            for ds_key, ds_meta in idx_data["datasets"].items():
                for art in ds_meta["artifacts"].values():
                    art["path"] = str(pathlib.Path(art["path"]).relative_to(tpath))
                for raw in ds_meta["raw_jsonl"].values():
                    raw["path"] = str(pathlib.Path(raw["path"]).relative_to(tpath))
            idx_p.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")

            # Clean relative index loads cleanly
            store = fics_showcase.ShowcaseStore(idx_p)
            self.assertIn("scope_a", store.datasets)

            # Tamper replay file without updating sha256 -> must fail!
            rep_rel = idx_data["datasets"]["scope_a"]["artifacts"]["replay_report"]["path"]
            rep_file = tpath / rep_rel
            rep_orig = rep_file.read_bytes()
            rep_file.write_bytes(rep_orig + b"\n")
            with self.assertRaises(ValueError):
                fics_showcase.ShowcaseStore(idx_p)
            rep_file.write_bytes(rep_orig)

            # Tamper raw_jsonl without updating sha256 -> must fail!
            raw_rel = idx_data["datasets"]["scope_a"]["raw_jsonl"]["qwen3"]["path"]
            raw_file = tpath / raw_rel
            raw_orig = raw_file.read_bytes()
            raw_file.write_bytes(raw_orig + b" ")
            with self.assertRaises(ValueError):
                fics_showcase.ShowcaseStore(idx_p)
            raw_file.write_bytes(raw_orig)


if __name__ == "__main__":
    unittest.main()
