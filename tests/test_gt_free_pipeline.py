"""CPU interface unit tests for the thin GT-free orchestration module."""

from __future__ import annotations

import json
import pathlib
import sys

from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pcb_vlm.evaluation.candidate_verifier import ALLOWED_SEVEN_LABELS, DEFAULT_VERIFIER_PROMPT  # noqa: E402
from pcb_vlm.evaluation.gt_free_pipeline import (  # noqa: E402
    compute_board_roi_and_tiles,
    export_verified_predictions_gt_free,
    prepare_verifier_manifest_gt_free,
    reconstruct_board_predictions_gt_free,
)
import hashlib  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402

from pcb_vlm.evaluation.gt_free_pipeline import load_jsonl_records  # noqa: E402


def _make_verifier_spec() -> dict:
    return {
        "model_key": "qwen3",
        "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "dtype": "bfloat16",
        "do_sample": False,
        "repetition_penalty": 1.0,
        "max_new_tokens": 32,
        "min_pixels": 65536,
        "max_pixels": 200704,
        "allowed_labels": list(ALLOWED_SEVEN_LABELS),
        "prompt": DEFAULT_VERIFIER_PROMPT,
        "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
    }


def test_gt_free_pipeline_end_to_end_cpu(tmp_path: pathlib.Path) -> None:
    import numpy as np

    board_path = tmp_path / "demo_board.png"
    arr = np.full((1200, 1400, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[100:1100, 100:1200]
    arr[100:1100, 100:1200, 0] = (xx % 200).astype(np.uint8)
    arr[100:1100, 100:1200, 1] = (yy % 200).astype(np.uint8)
    arr[100:1100, 100:1200, 2] = 60
    Image.fromarray(arr).save(board_path)

    geom = compute_board_roi_and_tiles(board_path, export_dir=tmp_path / "inputs", board_key="demo")
    assert geom["orig_wh"] == [1400, 1200]
    assert geom["tile_count"] >= 4
    assert pathlib.Path(geom["roi_export"]["path"]).is_file()

    # Fake tile T00 raw output -> global coords + NMS
    raw_records = [
        {
            "call_id": "qwen3_formal_demo_T00",
            "board_id": "demo",
            "mode": "T",
            "call_index": 0,
            "ok": True,
            "width": 1024,
            "height": 1024,
            "roi_offset_x": geom["roi_offset_x"],
            "roi_offset_y": geom["roi_offset_y"],
            "tile_offset_x": 0,
            "tile_offset_y": 0,
            "max_new_tokens": 512,
            "generated_tokens": 20,
            "generated_token_ids": [151645],
            "raw_text": json.dumps([
                {"bbox_2d": [100, 100, 200, 200], "label": "capacitor"},
                {"bbox_2d": [300, 300, 420, 420], "label": "resistor"},
                {"bbox_2d": [500, 500, 560, 560], "label": "diode"},
            ]) + "<|im_end|>",
        }
    ]

    recon = reconstruct_board_predictions_gt_free(
        board_id="demo",
        image_id=99,
        orig_wh=(1400, 1200),
        raw_records=raw_records,
        mode="T",
        expected_call_ids=["qwen3_formal_demo_T00"],
    )
    assert recon["technical_complete"] is True
    assert recon["post_nms_predictions"] == 3

    manifest, integrity = prepare_verifier_manifest_gt_free(
        reconstructed_by_board=[recon],
        board_image_paths={"demo": board_path},
        crops_dir=tmp_path / "crops",
        inference_spec=_make_verifier_spec(),
        source_provenance={"boards": ["demo"], "mode": "T", "model_key": "qwen3"},
    )
    assert integrity["unique_calls"] == 3
    assert integrity["total_predictions"] == 3
    assert manifest["counts"]["unique_calls"] == 3

    verifier_rows = [
        {"call_id": "candB_u0001", "ok": True, "raw_text": '{"label": "resistor"}<|im_end|>', "generated_tokens": 6, "normal_eos": True},
        {"call_id": "candB_u0002", "ok": True, "raw_text": '{"label": "other"}<|im_end|>', "generated_tokens": 6, "normal_eos": True},
        {"call_id": "candB_u0003", "ok": True, "raw_text": '{"label": "igbt"}<|im_end|>', "generated_tokens": 6, "normal_eos": True},
    ]
    out_json = tmp_path / "exported.json"
    exported = export_verified_predictions_gt_free(
        manifest=manifest,
        verifier_rows=verifier_rows,
        policy="B_other",
        output_path=out_json,
    )
    assert out_json.is_file()
    b_res = exported["per_board"]["demo"]
    assert b_res["kept_count"] == 1
    assert b_res["rejected_count"] == 2
    assert b_res["pred_record"]["objects"][0]["label"] == "capacitor"
    assert exported["summary"]["rejected_as_other"] == 1
    assert exported["summary"]["format_failure_filtered"] == 1
    assert exported["summary"]["cross_target_retained_same_orig_class"] == 1


def test_reconstruct_zero_candidates_and_incomplete_handling(tmp_path: pathlib.Path) -> None:
    rec_empty = [
        {
            "call_id": "qwen3_formal_b1_T00",
            "board_id": "b1",
            "mode": "T",
            "call_index": 0,
            "ok": True,
            "orig_input_width": 1024,
            "orig_input_height": 1024,
            "roi_offset_x": 0,
            "roi_offset_y": 0,
            "tile_offset_x": 0,
            "tile_offset_y": 0,
            "max_new_tokens": 512,
            "generated_tokens": 2,
            "generated_token_ids": [151645],
            "raw_output": "[]<|im_end|>",
        }
    ]
    res_ok = reconstruct_board_predictions_gt_free(
        board_id="b1",
        image_id=1,
        orig_wh=(1024, 1024),
        raw_records=rec_empty,
        expected_call_ids=["qwen3_formal_b1_T00"],
    )
    assert res_ok["technical_complete"] is True
    assert res_ok["predictions"] == []

    board_img = tmp_path / "b1.png"
    Image.new("RGB", (1024, 1024), (20, 40, 20)).save(board_img)
    manifest0, integrity0 = prepare_verifier_manifest_gt_free(
        reconstructed_by_board=[res_ok],
        board_image_paths={"b1": board_img},
        crops_dir=tmp_path / "crops0",
        inference_spec=_make_verifier_spec(),
        source_provenance={"boards": ["b1"], "mode": "T", "model_key": "qwen3"},
    )
    assert integrity0["unique_calls"] == 0
    exported0 = export_verified_predictions_gt_free(manifest=manifest0, verifier_rows=[], policy="B_other")
    assert exported0["technical_complete"] is True
    assert exported0["summary"]["total_candidates"] == 0

    res_inc = reconstruct_board_predictions_gt_free(
        board_id="b1",
        image_id=1,
        orig_wh=(1024, 1024),
        raw_records=rec_empty,
        expected_call_ids=["qwen3_formal_b1_T00", "qwen3_formal_b1_T01"],
    )
    assert res_inc["technical_complete"] is False
    assert res_inc["missing_call_ids"] == ["qwen3_formal_b1_T01"]


def test_reconstruct_w_skips_nms_while_t_applies_nms() -> None:
    overlapping = json.dumps([
        {"bbox_2d": [100, 100, 300, 300], "label": "resistor"},
        {"bbox_2d": [105, 105, 295, 295], "label": "resistor"},
    ]) + "<|im_end|>"
    rec_w = [{
        "call_id": "qwen3_formal_b_W",
        "board_id": "b",
        "mode": "W",
        "call_index": 0,
        "ok": True,
        "orig_input_width": 1000,
        "orig_input_height": 1000,
        "roi_offset_x": 0,
        "roi_offset_y": 0,
        "tile_offset_x": 0,
        "tile_offset_y": 0,
        "max_new_tokens": 512,
        "generated_tokens": 16,
        "generated_token_ids": [151645],
        "raw_output": overlapping,
    }]
    out_w = reconstruct_board_predictions_gt_free(
        board_id="b",
        image_id=7,
        orig_wh=(1000, 1000),
        raw_records=rec_w,
        mode="W",
        expected_call_ids=["qwen3_formal_b_W"],
    )
    assert out_w["technical_complete"] is True
    assert out_w["nms_applied"] is False
    assert len(out_w["predictions"]) == 2

    rec_t = [dict(rec_w[0], call_id="qwen3_formal_b_T00", mode="T")]
    out_t = reconstruct_board_predictions_gt_free(
        board_id="b",
        image_id=7,
        orig_wh=(1000, 1000),
        raw_records=rec_t,
        mode="T",
        expected_call_ids=["qwen3_formal_b_T00"],
    )
    assert out_t["technical_complete"] is True
    assert out_t["nms_applied"] is True
    assert len(out_t["predictions"]) == 1

    with pytest.raises(ValueError):
        reconstruct_board_predictions_gt_free(
            board_id="b",
            image_id=7,
            orig_wh=(1000, 1000),
            raw_records=rec_w,
            mode="Z",
        )


def test_reconstruct_and_export_reject_duplicate_and_unexpected_call_ids(tmp_path: pathlib.Path) -> None:
    base_rec = {
        "call_id": "qwen3_formal_b1_T00",
        "board_id": "b1",
        "mode": "T",
        "call_index": 0,
        "ok": True,
        "orig_input_width": 1000,
        "orig_input_height": 1000,
        "roi_offset_x": 0,
        "roi_offset_y": 0,
        "tile_offset_x": 0,
        "tile_offset_y": 0,
        "max_new_tokens": 512,
        "generated_tokens": 12,
        "generated_token_ids": [151645],
        "raw_output": '[{"bbox_2d": [100, 100, 200, 200], "label": "ic"}]<|im_end|>',
    }
    dup_res = reconstruct_board_predictions_gt_free(
        board_id="b1",
        image_id=1,
        orig_wh=(1000, 1000),
        raw_records=[base_rec, dict(base_rec)],
        mode="T",
        expected_call_ids=["qwen3_formal_b1_T00"],
    )
    assert dup_res["technical_complete"] is False
    assert dup_res["duplicate_call_ids"] == ["qwen3_formal_b1_T00"]

    extra_rec = dict(base_rec, call_id="qwen3_formal_b1_T99")
    extra_res = reconstruct_board_predictions_gt_free(
        board_id="b1",
        image_id=1,
        orig_wh=(1000, 1000),
        raw_records=[base_rec, extra_rec],
        mode="T",
        expected_call_ids=["qwen3_formal_b1_T00"],
    )
    assert extra_res["technical_complete"] is False
    assert extra_res["unexpected_call_ids"] == ["qwen3_formal_b1_T99"]

    valid_res = reconstruct_board_predictions_gt_free(
        board_id="b1",
        image_id=1,
        orig_wh=(1000, 1000),
        raw_records=[base_rec],
        mode="T",
        expected_call_ids=["qwen3_formal_b1_T00"],
    )
    board_img = tmp_path / "b1.png"
    Image.new("RGB", (1000, 1000), (30, 50, 30)).save(board_img)
    manifest, _ = prepare_verifier_manifest_gt_free(
        reconstructed_by_board=[valid_res],
        board_image_paths={"b1": board_img},
        crops_dir=tmp_path / "crops",
        inference_spec=_make_verifier_spec(),
        source_provenance={"boards": ["b1"], "mode": "T", "model_key": "qwen3"},
    )
    with pytest.raises(RuntimeError, match="duplicate/unexpected"):
        export_verified_predictions_gt_free(
            manifest=manifest,
            verifier_rows=[
                {"call_id": "candB_u0001", "ok": True, "raw_text": '{"label": "ic"}<|im_end|>'},
                {"call_id": "candB_u0001", "ok": True, "raw_text": '{"label": "other"}<|im_end|>'},
            ],
            policy="B_other",
        )
    with pytest.raises(RuntimeError, match="duplicate/unexpected"):
        export_verified_predictions_gt_free(
            manifest=manifest,
            verifier_rows=[
                {"call_id": "candB_u0001", "ok": True, "raw_text": '{"label": "ic"}<|im_end|>'},
                {"call_id": "candB_u9999", "ok": True, "raw_text": '{"label": "ic"}<|im_end|>'},
            ],
            policy="B_other",
        )


def test_heldout_reconstruction_and_b_other_regression_cpu() -> None:
    fixture_dir_str = os.environ.get("FICS_HELDOUT_ARTIFACT_DIR")
    if not fixture_dir_str:
        pytest.skip("Set FICS_HELDOUT_ARTIFACT_DIR to run held-out CPU reconstruction regression")
    fixture_dir = pathlib.Path(fixture_dir_str)
    raw_q3 = load_jsonl_records(fixture_dir / "baseline_runs" / "run_qwen3" / "raw_outputs_all.jsonl")
    manifest = json.loads((fixture_dir / "heldout_candidate_to_crop_manifest.json").read_text(encoding="utf-8"))
    verifier_rows = load_jsonl_records(fixture_dir / "verifier_run" / "verifier_raw_outputs.jsonl")
    ref_b_other = json.loads((fixture_dir / "heldout_b_other_predictions_gt_free.json").read_text(encoding="utf-8"))

    reconstructed_all = []
    for bid, img_id in (("s19", 19), ("s16", 16)):
        expected_ids = [f"qwen3_formal_{bid}_T{i:02d}" for i in range(70)]
        recon = reconstruct_board_predictions_gt_free(
            board_id=bid,
            image_id=img_id,
            orig_wh=(8256, 5504),
            raw_records=raw_q3,
            mode="T",
            expected_call_ids=expected_ids,
        )
        assert recon["technical_complete"] is True
        assert recon["post_nms_predictions"] == 254
        reconstructed_all.extend(recon["predictions"])

    assert len(reconstructed_all) == 508
    assert [(p["board_id"], p["label"], p["bbox_xyxy"]) for p in reconstructed_all] == [
        (p["board_id"], p["orig_pred_label"], p["orig_pred_bbox_xyxy"]) for p in manifest["predictions"]
    ]

    exported = export_verified_predictions_gt_free(
        manifest=manifest,
        verifier_rows=verifier_rows,
        policy="B_other",
    )
    assert exported["summary"] == ref_b_other["summary"]
    assert exported["summary"]["retained_count"] == 150
    for bid in ("s16", "s19"):
        assert exported["per_board"][bid]["pred_record"] == ref_b_other["per_board"][bid]["pred_record"]
