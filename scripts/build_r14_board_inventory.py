"""Build CPU-only physical-board & SHA-256 inventory for next-round FICS adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402

_RAW_TYPE_TO_FICS6 = {
    "capacitor": "capacitor", "capacitors": "capacitor",
    "resistor": "resistor", "resistors": "resistor",
    "inductor": "inductor", "inductors": "inductor",
    "transistor": "transistor", "transistors": "transistor",
    "diode": "diode", "diodes": "diode",
    "ic": "ic", "ics": "ic",
}


def _count_csv_fics6(csv_path: pathlib.Path) -> Dict[str, Any]:
    counts = {c: 0 for c in FICS_SIX_CLASSES}
    non_fics: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_t = str(row.get("component_type") or "").strip().lower()
            mapped = _RAW_TYPE_TO_FICS6.get(raw_t)
            if mapped:
                counts[mapped] += 1
            elif raw_t:
                non_fics[raw_t] = non_fics.get(raw_t, 0) + 1
    return {"fics6_total": sum(counts.values()), "class_counts": counts, "dropped_non_6class": non_fics}


def _count_coco_fics6(coco_path: pathlib.Path, image_id: Optional[int] = None) -> Dict[str, Any]:
    doc = json.loads(coco_path.read_text(encoding="utf-8"))
    cat_map = {int(c["id"]): str(c["name"]).strip().lower() for c in doc.get("categories") or []}
    images = list(doc.get("images") or [])
    tid = int(image_id) if image_id is not None else int(images[0]["id"])
    counts = {c: 0 for c in FICS_SIX_CLASSES}
    for ann in doc.get("annotations") or []:
        if int(ann["image_id"]) != tid:
            continue
        lbl = cat_map.get(int(ann["category_id"]), "")
        if lbl in counts:
            counts[lbl] += 1
    return {"fics6_total": sum(counts.values()), "class_counts": counts, "dropped_non_6class": {}}


BOARD_CATALOG: List[Dict[str, Any]] = [
    # Train pool (7 physical boards, 8 sides)
    {
        "board_side_id": "s18_front", "physical_board": "s18", "split": "train_pool",
        "family_note": "Historical dev board (R8 SFT train)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-s18-audit-20261114\extracted\s18\DSLR\img\s18_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-s18-audit-20261114\extracted\s18\DSLR\annotation\s18_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-s18-audit-20261114\sample_coco_s18.json",
    },
    {
        "board_side_id": "s27_front", "physical_board": "s27", "split": "train_pool",
        "family_note": "Historical unobserved dev board (R8 SFT train)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-validation-20261115\extracted\s27\s27\DSLR\img\s27_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-validation-20261115\extracted\s27\s27\DSLR\annotation\s27_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-validation-20261115\coco\s27_front_coco.json",
    },
    {
        "board_side_id": "s28_front", "physical_board": "s28", "split": "train_pool",
        "family_note": "Historical dev board (R8 SFT train)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-qwen3-s28-dev-20261114\extracted\s28\DSLR\img\s28_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-qwen3-s28-dev-20261114\extracted\s28\DSLR\annotation\s28_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-qwen3-s28-dev-20261114\selected_coco_s28_front.json",
    },
    {
        "board_side_id": "s29_front", "physical_board": "s29", "split": "train_pool",
        "family_note": "Historical unobserved dev board (R8 SFT train)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-resume-20261115\extracted\s29\s29\DSLR\img\s29_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-resume-20261115\extracted\s29\s29\DSLR\annotation\s29_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-fics-unobserved-resume-20261115\coco\s29_front_coco.json",
    },
    {
        "board_side_id": "s8_front", "physical_board": "s8", "split": "train_pool",
        "family_note": "Compact Lattice LCMX0256C board (front+back grouped in train_pool)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s8\DSLR\img\s8_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s8\DSLR\annotation\s8_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s8_front_official_coco.json",
    },
    {
        "board_side_id": "s8_back", "physical_board": "s8", "split": "train_pool",
        "family_note": "Compact Lattice LCMX0256C board (front+back grouped in train_pool)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s8\DSLR\img\s8_back.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s8\DSLR\annotation\s8_back.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s8_back_official_coco.json",
    },
    {
        "board_side_id": "s21_front", "physical_board": "s21", "split": "train_pool",
        "family_note": "IBM36 / AMSRC04 dense board (230 GT)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s21\DSLR\img\s21_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s21\DSLR\annotation\s21_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s21_front_official_coco.json",
    },
    {
        "board_side_id": "s22_front", "physical_board": "s22", "split": "train_pool",
        "family_note": "Dense 218-GT board (14 transistors, 97 resistors)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s22\DSLR\img\s22_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s22\DSLR\annotation\s22_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s22_front_official_coco.json",
    },
    {
        "board_side_id": "s30_front", "physical_board": "s30", "split": "train_pool",
        "family_note": "Samsung DRAM / YDA912ECC dense board (225 GT)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s30\DSLR\img\s30_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s30\DSLR\annotation\s30_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s30_front_official_coco.json",
    },
    # Validation split (isolated from train & reserved blind; grouped by physical board)
    {
        "board_side_id": "s3_front", "physical_board": "s3", "split": "val_split",
        "family_note": "Cisco WIC family (grouped into val_split; kept separate from train_pool)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s3\DSLR\img\s3_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s3\DSLR\annotation\s3_front.csv",
        "coco_path": None,
    },
    {
        "board_side_id": "s2_front", "physical_board": "s2", "split": "val_split",
        "family_note": "Conexant/IDT/PI74FCT green carrier + Cisco WIC vendor proximity (front+back grouped in val_split)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s2\DSLR\img\s2_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s2\DSLR\annotation\s2_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s2_front_official_coco.json",
    },
    {
        "board_side_id": "s2_back", "physical_board": "s2", "split": "val_split",
        "family_note": "Conexant/IDT/PI74FCT green carrier (front+back grouped in val_split)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s2\DSLR\img\s2_back.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s2\DSLR\annotation\s2_back.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s2_back_official_coco.json",
    },
    # R14 Dev Evaluation Boards (strictly dev, NOT blind)
    {
        "board_side_id": "s20_front", "physical_board": "s20", "split": "dev_eval_r14",
        "family_note": "R14 3-way dev comparison board (111 GT; used in R1-R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\annotation\s20_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s20_front_coco.json",
    },
    {
        "board_side_id": "s4_front", "physical_board": "s4", "split": "dev_eval_r14",
        "family_note": "R14 3-way dev comparison board (52 GT; used in R2-R5, R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\annotation\s4_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s4_front_coco.json",
    },
    # R14 Unused Real Photo Smoke Only (excluded from train & blind due to Cisco proximity to s2/s3)
    {
        "board_side_id": "s1_front", "physical_board": "s1", "split": "smoke_only_unused_r14",
        "family_note": "Cisco WIC family (unused prior to R14; used ONLY for R14 live upload-to-display smoke)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\img\s1_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\annotation\s1_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s1_front_official_coco.json",
    },
    # Untouched Reserved Blind Boards (ZERO inference, ZERO tuning in R14)
    {
        "board_side_id": "s31_front", "physical_board": "s31", "split": "reserved_untouched_blind",
        "family_note": "Designated next-round untouched FICS adaptation blind board (ZERO GPU inference in R1-R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s31\DSLR\img\s31_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s31\DSLR\annotation\s31_front.csv",
        "coco_path": None,
    },
    {
        "board_side_id": "s10_front", "physical_board": "s10", "split": "reserved_untouched_blind",
        "family_note": "Blue VGA carrier family — Codex independent pure-VLM blind benchmark (ZERO tuning/touch in R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s10\DSLR\img\s10_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s10\DSLR\annotation\s10_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s10_front_official_coco.json",
    },
    {
        "board_side_id": "s16_front", "physical_board": "s16", "split": "reserved_untouched_blind",
        "family_note": "Sealed held-out board (ZERO new GPU inference in R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s16\DSLR\img\s16_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s16\DSLR\annotation\s16_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-heldout-evaluation-20261116\coco\s16_front_coco.json",
    },
    {
        "board_side_id": "s19_front", "physical_board": "s19", "split": "reserved_untouched_blind",
        "family_note": "Sealed held-out board (ZERO new GPU inference in R14)",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s19\DSLR\img\s19_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s19\DSLR\annotation\s19_front.csv",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-heldout-evaluation-20261116\coco\s19_front_coco.json",
    },
    # Excluded WACV train overlap
    {
        "board_side_id": "s9_front", "physical_board": "s9", "split": "excluded_wacv_overlap",
        "family_note": "Digilent Zybo board — Zybo exists in WACV YOLO11n training split (independent-data-audit.json), EXCLUDED from adaptation & validation",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s9\DSLR\img\s9_front.tif",
        "csv_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s9\DSLR\annotation\s9_front.csv",
        "coco_path": None,
    },
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build physical-board & SHA-256 inventory for next-round FICS adaptation.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    records: List[Dict[str, Any]] = []
    by_split_boards: Dict[str, set[str]] = {}
    by_split_shas: Dict[str, set[str]] = {}
    split_totals: Dict[str, Dict[str, int]] = {}

    for entry in BOARD_CATALOG:
        tp = pathlib.Path(entry["tif_path"])
        cp = pathlib.Path(entry["csv_path"])
        cop = pathlib.Path(entry["coco_path"]) if entry["coco_path"] else None
        if not tp.is_file() or not cp.is_file():
            raise FileNotFoundError(f"Missing board assets for {entry['board_side_id']}: tif={tp.is_file()}, csv={cp.is_file()}")
        t_sha = file_sha256(tp).lower()
        c_sha = file_sha256(cp).lower()
        co_sha = file_sha256(cop).lower() if (cop and cop.is_file()) else None
        counts_info = _count_coco_fics6(cop) if (cop and cop.is_file()) else _count_csv_fics6(cp)

        sp = str(entry["split"])
        pb = str(entry["physical_board"])
        by_split_boards.setdefault(sp, set()).add(pb)
        by_split_shas.setdefault(sp, set()).add(t_sha)
        st = split_totals.setdefault(sp, {"sides": 0, "physical_boards": 0, "gt_total": 0})
        st["sides"] += 1
        st["gt_total"] += int(counts_info["fics6_total"])

        records.append({
            "board_side_id": entry["board_side_id"],
            "physical_board": pb,
            "split": sp,
            "family_note": entry["family_note"],
            "tif_path": str(tp),
            "tif_sha256": t_sha,
            "csv_path": str(cp),
            "csv_sha256": c_sha,
            "coco_path": str(cop) if cop else None,
            "coco_sha256": co_sha,
            "fics6_gt_total": counts_info["fics6_total"],
            "class_counts": counts_info["class_counts"],
            "dropped_non_6class": counts_info["dropped_non_6class"],
        })

    for sp, st in split_totals.items():
        st["physical_boards"] = len(by_split_boards[sp])

    # Cross-split leakage checks
    splits = list(by_split_boards.keys())
    board_overlaps = []
    sha_overlaps = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            b_int = sorted(by_split_boards[s1] & by_split_boards[s2])
            h_int = sorted(by_split_shas[s1] & by_split_shas[s2])
            if b_int:
                board_overlaps.append({"splits": [s1, s2], "boards": b_int})
            if h_int:
                sha_overlaps.append({"splits": [s1, s2], "sha256s": h_int})

    doc = {
        "schema_version": "r14_fics_adaptation_board_inventory_v1",
        "wacv_overlap_exclusion_note": "S9 (Digilent Zybo) is excluded from FICS adaptation & validation because 'Zybo' appears in WACV YOLO11n 21-board training set (independent-data-audit.json).",
        "split_summary": split_totals,
        "isolation_audit": {
            "zero_physical_board_overlap": len(board_overlaps) == 0,
            "zero_image_sha256_overlap": len(sha_overlaps) == 0,
            "board_overlaps": board_overlaps,
            "sha_overlaps": sha_overlaps,
            "reserved_untouched_blind_boards": sorted(by_split_boards.get("reserved_untouched_blind", set())),
            "zero_gpu_inference_in_r14_on_reserved": True,
        },
        "boards": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"split_summary": split_totals, "isolation_audit": doc["isolation_audit"]}, indent=2, ensure_ascii=False))
    assert len(board_overlaps) == 0 and len(sha_overlaps) == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
