"""Round 16 Step 3: Official archive weak-class (diode/transistor/IC) & physical-family inventory (CPU only).

Strict invariants:
- Zero GPU calls and zero inference/preview on newly designated blind acceptance boards.
- Explicitly bans `s31` (already publicly evaluated), `s9` (WACV Zybo overlap), `s20`, `s4`, `s1`, and `s10`/`s16`/`s19`.
- Groups remaining unused official FICS-PCB DSLR boards by physical family (vendor/chip/text/back-front pairing)
  into strictly disjoint Next-Round Training Expansion Candidates vs Independent New-Family Blind Acceptance Candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile
from typing import Any, Dict, List, Optional, Sequence

PROBE_JSON = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\fics_all_candidates_probe.json")
R14_INV = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json")
FPIC_ZIP = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-dataset-alignment-20261114\fpic_official\smd_annotation.zip")
WACV_ZIP = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019.zip")
DEFAULT_OUT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124\data_audit\r16_weak_class_family_inventory.json")

# Physical family & usage classification for all 31 FICS-PCB boards (s1..s31)
FAMILY_MAP: Dict[str, Dict[str, str]] = {
    "s1":  {"family": "cisco_wic_smd_v1", "status": "r14_r15_dev_extra"},
    "s2":  {"family": "conexant_idt_pi74fct_carrier", "status": "r15_val_split"},
    "s3":  {"family": "cisco_wic_idt_pi74fct_v1", "status": "r15_val_split"},
    "s4":  {"family": "adesio_ti_tps53318_board", "status": "r14_r15_dev_eval"},
    "s5":  {"family": "altera_epm7128_ds2172_idt_sister_A", "status": "unused_unfetched"},
    "s6":  {"family": "altera_epm7128_ds2172_idt_sister_A", "status": "unused_unfetched"},
    "s7":  {"family": "altera_epm7128_ds2172_idt_sister_A", "status": "unused_unfetched"},
    "s8":  {"family": "lattice_lcmxo256c_compact", "status": "r15_train_pool"},
    "s9":  {"family": "digilent_zybo_zynq_xc7z010", "status": "excluded_wacv_zybo_overlap"},
    "s10": {"family": "blue_vga_carrier_1117", "status": "reserved_blind_codex_pure_vlm"},
    "s16": {"family": "numato_mimas_v2_spartan6_xc6slx9", "status": "reserved_heldout_blind"},
    "s17": {"family": "ti_msp430_launchpad_family", "status": "unused_csv_probed_only_same_family_as_s20"},
    "s18": {"family": "usi_bm09s_module_board", "status": "r15_train_pool"},
    "s19": {"family": "ti_tpa3110_audio_amp_board", "status": "reserved_heldout_blind"},
    "s20": {"family": "ti_msp430_launchpad_family", "status": "r14_r15_dev_eval"},
    "s21": {"family": "ibm36_amsrc04_dense", "status": "r15_train_pool"},
    "s22": {"family": "dense_220gt_multi_transistor", "status": "r15_train_pool"},
    "s23": {"family": "tdk_dense_77cap_87res", "status": "unused_csv_probed_only"},
    "s24": {"family": "dense_161gt_cap_res_ic", "status": "unused_csv_probed_only"},
    "s25": {"family": "hitachi_lsi_dram_305gt", "status": "unused_csv_probed_only"},
    "s27": {"family": "r8_dev_family_s27_s28_s29", "status": "r15_train_pool"},
    "s28": {"family": "r8_dev_family_s27_s28_s29", "status": "r15_train_pool"},
    "s29": {"family": "r8_dev_family_s27_s28_s29", "status": "r15_train_pool"},
    "s30": {"family": "samsung_dram_yda912ecc", "status": "r15_train_pool"},
    "s31": {"family": "numato_elbert_v2_spartan3_xc3s50a", "status": "s31_public_evaluated_banned_from_train_cal"},
}


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest().lower()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit weak-class & physical-family candidates across official FICS-PCB.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    r14 = json.loads(R14_INV.read_text(encoding="utf-8"))
    r14_by_side = {b["board_side_id"]: b for b in r14["boards"]}

    probed_entries: List[Dict[str, Any]] = []
    for c in probe["candidates"]:
        bid = str(c["board_id"])
        fam_info = FAMILY_MAP.get(bid, {"family": f"unknown_{bid}", "status": "unused_unfetched"})
        img_entries = {e["basename"]: e for e in (c.get("dslr_img_entries") or []) if not e["basename"].startswith(".")}
        for a in c.get("dslr_annotations") or []:
            stem = pathlib.Path(a["csv_name"]).stem
            tif_name = f"{stem}.tif"
            ct = a.get("component_types") or {}
            counts6 = {
                "capacitor": int(ct.get("capacitors", 0)),
                "resistor": int(ct.get("resistors", 0)),
                "inductor": int(ct.get("inductors", 0)),
                "transistor": int(ct.get("transistors", 0)),
                "diode": int(ct.get("diodes", 0)),
                "ic": int(ct.get("ICs", 0)),
            }
            r14_rec = r14_by_side.get(stem)
            probed_entries.append({
                "board_side_id": stem,
                "physical_board": bid,
                "physical_family": fam_info["family"],
                "current_status": fam_info["status"],
                "official_zip_url": c["url"],
                "official_etag": c["etag"],
                "csv_name": a["csv_name"],
                "csv_sha256": a["csv_sha256"],
                "tif_name": tif_name,
                "tif_crc32_hex": (img_entries.get(tif_name) or {}).get("crc32_hex"),
                "tif_file_mb": (img_entries.get(tif_name) or {}).get("file_mb"),
                "local_tif_present": bool(r14_rec and r14_rec.get("tif_path") and pathlib.Path(r14_rec["tif_path"]).is_file()),
                "local_tif_sha256": r14_rec.get("tif_sha256") if r14_rec else None,
                "fics6_counts": counts6,
                "weak_class_sum_diode_transistor": counts6["diode"] + counts6["transistor"],
                "sample_texts": (a.get("sample_texts") or [])[:6],
            })

    # Also summarize local external datasets (FPIC SMD & WACV)
    fpic_info = {"path": str(FPIC_ZIP), "exists": FPIC_ZIP.is_file(), "sha256": _sha256(FPIC_ZIP) if FPIC_ZIP.is_file() else None}
    if FPIC_ZIP.is_file():
        with zipfile.ZipFile(FPIC_ZIP, "r") as zf:
            fpic_info["zip_entry_count"] = len(zf.namelist())
    wacv_info = {"path": str(WACV_ZIP), "exists": WACV_ZIP.is_file(), "sha256": _sha256(WACV_ZIP) if WACV_ZIP.is_file() else None}

    # Current R15 train_pool weak class totals
    r15_train_boards = [b for b in r14["boards"] if b["split"] == "train_pool"]
    r15_train_totals = {
        c: sum(int(b["class_counts"][c]) for b in r15_train_boards)
        for c in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic")
    }

    # Build strictly physical-family-disjoint candidate lists from unused_unfetched / unused_csv_probed_only
    # Exclude s17 (same TI MSP430 family as dev S20) and s9_back (WACV Zybo overlap).
    # Note s4_back is same physical board as dev s4_front -> excluded from training AND new-family blind!
    # Note s16_back is same physical board as heldout s16_front -> keep reserved with s16!
    train_expansion_candidates = [
        {
            "board_side_id": "s25_front",
            "physical_board": "s25",
            "physical_family": "hitachi_lsi_dram_305gt",
            "reason": "High IC diversity (22 ICs + 2 diodes) on a Hitachi/LSI DRAM board; completely distinct from dev/val/S31 families.",
            "fics6_counts": next(x["fics6_counts"] for x in probed_entries if x["board_side_id"] == "s25_front"),
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s25_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s25_front"),
        },
        {
            "board_side_id": "s23_front",
            "physical_board": "s23",
            "physical_family": "tdk_dense_77cap_87res",
            "reason": "Adds 8 ICs + 1 transistor + dense 173-GT layout; distinct physical family.",
            "fics6_counts": next(x["fics6_counts"] for x in probed_entries if x["board_side_id"] == "s23_front"),
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s23_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s23_front"),
        },
        {
            "board_side_id": "s5_front",
            "physical_board": "s5",
            "physical_family": "altera_epm7128_ds2172_idt_sister_A",
            "reason": "Adds 9 ICs (Altera MAX EPM7128) + 1 inductor; sister boards s6/s7 should stay in the same split or be omitted to avoid sister-board duplication.",
            "fics6_counts": next(x["fics6_counts"] for x in probed_entries if x["board_side_id"] == "s5_front"),
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s5_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s5_front"),
        },
        {
            "board_side_id": "s5_back",
            "physical_board": "s5",
            "physical_family": "altera_epm7128_ds2172_idt_sister_A",
            "reason": "Back side of s5 adding 12 ICs; grouped with s5_front.",
            "fics6_counts": next(x["fics6_counts"] for x in probed_entries if x["board_side_id"] == "s5_back"),
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s5_back"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s5_back"),
        },
    ]

    # Independent New-Family Blind Acceptance Candidates (zero GPU inference or preview in R16!)
    # Note: In the remaining unused FICS boards, s17 has 6 diodes + 2 transistors + 8 ICs (note: TI MSP430 LaunchPad family proximity to s20_front, so note that explicitly!),
    # while s10 (7 diodes, 3 transistors, 7 ICs, blue VGA carrier) and s16 (9 diodes, 1 transistor, 4 ICs, Numato Mimas V2) are already held-out blind with distinct families.
    # For a brand-new unfetched board: s25/s23/s5 have few diodes/transistors; s17 has rich diode+transistor+IC (with MSP430 family note), and s11..s15/s26 remain unprobed on Trust-Hub.
    independent_blind_candidates = [
        {
            "board_side_id": "s17_front",
            "physical_board": "s17",
            "physical_family": "ti_msp430_launchpad_family",
            "weak_class_presence": {"diode": 6, "transistor": 2, "ic": 8, "inductor": 3},
            "family_caveat": "Shares TI MSP430 silkscreen family with dev board s20 (though distinct board PCB); zero GPU inference or image preview performed in R16.",
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s17_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s17_front"),
            "r16_gpu_calls": 0,
        },
        {
            "board_side_id": "s10_front",
            "physical_board": "s10",
            "physical_family": "blue_vga_carrier_1117",
            "weak_class_presence": {"diode": 7, "transistor": 3, "ic": 7, "inductor": 12},
            "family_caveat": "Already locally extracted & untouched by YOLO training/calibration (used only in pure-VLM Qwen3 R11 audit); completely distinct blue VGA carrier family.",
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s10_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s10_front"),
            "r16_gpu_calls": 0,
        },
        {
            "board_side_id": "s16_front",
            "physical_board": "s16",
            "physical_family": "numato_mimas_v2_spartan6_xc6slx9",
            "weak_class_presence": {"diode": 9, "transistor": 1, "ic": 4, "inductor": 0},
            "family_caveat": "Held-out blind board (distinct Spartan-6 Mimas V2 board vs Elbert V2 s31); zero YOLO calls.",
            "csv_sha256": next(x["csv_sha256"] for x in probed_entries if x["board_side_id"] == "s16_front"),
            "official_zip_url": next(x["official_zip_url"] for x in probed_entries if x["board_side_id"] == "s16_front"),
            "r16_gpu_calls": 0,
        },
    ]

    report = {
        "schema_version": "r16_weak_class_family_inventory_v1",
        "r15_train_pool_class_counts": r15_train_totals,
        "external_local_archives": {
            "fpic_smd_annotation_zip": fpic_info,
            "wacv_2019_zip": wacv_info,
        },
        "unprobed_fics_ids_note": "s11..s15 and s26 were absent from fics_all_candidates_probe.json (either HTTP 404 or not in Trust-Hub DSLR batch); can be probed later if additional diode/transistor boards are needed.",
        "probed_fics_dslr_catalog": probed_entries,
        "next_round_training_expansion_candidates": train_expansion_candidates,
        "independent_new_family_blind_candidates": independent_blind_candidates,
        "banned_from_training_or_calibration": ["s31", "s9", "s20", "s4", "s1", "s10", "s16", "s19"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "r15_train_pool_class_counts": r15_train_totals,
        "probed_board_sides_total": len(probed_entries),
        "train_expansion_candidates": [x["board_side_id"] for x in train_expansion_candidates],
        "independent_blind_candidates": [x["board_side_id"] for x in independent_blind_candidates],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
