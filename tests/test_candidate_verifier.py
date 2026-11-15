"""Key CPU verification tests for GT-free second-stage candidate verifier and policies."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest
from PIL import Image

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.candidate_verifier import (
    DEFAULT_VERIFIER_PROMPT,
    apply_verifier_filter_policy,
    build_candidate_verifier_manifest,
    parse_single_label_json,
    verify_verifier_manifest_integrity,
)
from pcb_vlm.inference.candidate_verifier_runner import run_verifier_supervised

_BASE_SPEC = {
    "model_key": "qwen3",
    "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
    "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
    "model_local_dir": "/tmp/model",
    "overlay_dir": "/tmp/overlay",
    "dtype": "bfloat16",
    "do_sample": False,
    "repetition_penalty": 1.0,
    "max_new_tokens": 32,
    "min_pixels": 65536,
    "max_pixels": 200704,
    "allowed_labels": ["capacitor", "resistor", "inductor", "transistor", "diode", "ic", "other"],
    "prompt": DEFAULT_VERIFIER_PROMPT,
    "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
}


def _make_board(path: pathlib.Path, w: int = 800, h: int = 600) -> pathlib.Path:
    im = Image.new("RGB", (w, h), color=(20, 110, 50))
    for x in range(100, 180):
        for y in range(100, 160):
            im.putpixel((x, y), (200, 180, 60))
    im.save(path)
    return path


def test_gt_free_manifest_dedup_and_mapping_completeness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        b1 = _make_board(root / "board_a.tif")
        # Three predictions: p1 & p2 share exact same crop box -> exact SHA dedup to 1 unique call; p3 differs
        preds = [
            {"prediction_id": "p1", "board_id": "b1", "image_id": 1, "orig_order": 0, "label": "capacitor", "bbox_xyxy": [100.0, 100.0, 160.0, 150.0], "score": 1.0},
            {"prediction_id": "p2", "board_id": "b1", "image_id": 1, "orig_order": 1, "label": "resistor", "bbox_xyxy": [100.0, 100.0, 160.0, 150.0], "score": 1.0},
            {"prediction_id": "p3", "board_id": "b1", "image_id": 1, "orig_order": 2, "label": "ic", "bbox_xyxy": [200.0, 200.0, 320.0, 320.0], "score": 1.0},
        ]
        manifest = build_candidate_verifier_manifest(
            source_predictions=preds,
            board_image_paths={"b1": b1},
            crops_dir=root / "crops",
            inference_spec=_BASE_SPEC,
            source_provenance={"note": "unit_test_no_gt"},
        )
        assert manifest["counts"]["total_predictions"] == 3
        assert manifest["counts"]["unique_calls"] == 2
        assert manifest["counts"]["deduplicated_reused_predictions"] == 1
        assert manifest["predictions"][0]["unique_call_id"] == manifest["predictions"][1]["unique_call_id"]
        verified = verify_verifier_manifest_integrity(manifest)
        assert verified["total_predictions"] == 3
        assert verified["unique_calls"] == 2
        for u in manifest["unique_calls"]:
            assert "board_id" not in u and "orig_pred_label" not in u


def test_b_other_vs_b_exact_and_preserve_boxes_order() -> None:
    preds = [
        {"prediction_id": "p1", "board_id": "b1", "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [1.0, 2.0, 30.0, 40.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u1"},
        {"prediction_id": "p2", "board_id": "b1", "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [50.0, 60.0, 90.0, 95.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u2"},
        {"prediction_id": "p3", "board_id": "b1", "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u3"},
    ]
    records = {
        "u1": {"call_id": "u1", "ok": True, "strict_json_ok": True, "strict_label": "resistor"},   # cross-class target
        "u2": {"call_id": "u2", "ok": True, "strict_json_ok": True, "strict_label": "other"},      # background
        "u3": {"call_id": "u3", "ok": True, "strict_json_ok": True, "strict_label": "resistor"},   # exact match
    }
    res_other = apply_verifier_filter_policy(predictions=preds, verification_records_by_call_id=records, policy="B_other")
    res_exact = apply_verifier_filter_policy(predictions=preds, verification_records_by_call_id=records, policy="B_exact")

    # B_other keeps p1 (retaining orig label 'capacitor' and exact bbox!) and p3
    assert res_other["counts"]["retained_count"] == 2
    assert res_other["counts"]["rejected_as_other"] == 1
    assert res_other["kept_predictions_by_board"]["b1"] == [
        {"label": "capacitor", "bbox_xyxy": [1.0, 2.0, 30.0, 40.0], "score": 1.0},
        {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0},
    ]

    # B_exact drops p1 & p2 and keeps only exact match p3
    assert res_exact["counts"]["retained_count"] == 1
    assert res_exact["counts"]["cross_target_rejected"] == 1
    assert res_exact["counts"]["rejected_as_other"] == 1
    assert res_exact["kept_predictions_by_board"]["b1"] == [
        {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0},
    ]


def test_format_and_tech_failures_separated_and_compat_isolated() -> None:
    parsed = parse_single_label_json('I think {"label": "ic"} looks right')
    assert parsed["strict_json_ok"] is False
    assert parsed["strict_label"] is None
    assert parsed["compat_regex_label"] == "ic"

    preds = [
        {"prediction_id": "p1", "board_id": "b1", "orig_pred_label": "ic", "orig_pred_bbox_xyxy": [10, 10, 50, 50], "score": 1.0, "crop_valid": True, "unique_call_id": "u1"},
        {"prediction_id": "p2", "board_id": "b1", "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [20, 20, 60, 60], "score": 1.0, "crop_valid": True, "unique_call_id": "u2"},
    ]
    recs = {
        "u1": {"call_id": "u1", "ok": True, **parsed},
        "u2": {"call_id": "u2", "ok": False, "strict_json_ok": False, "strict_label": None},
    }
    res = apply_verifier_filter_policy(predictions=preds, verification_records_by_call_id=recs, policy="B_other")
    assert res["technical_complete"] is False
    assert res["counts"]["retained_count"] == 0
    assert res["counts"]["format_failure_filtered"] == 1
    assert res["counts"]["technical_failure_filtered"] == 1
    assert res["counts"]["rejected_as_other"] == 0


def test_supervised_gate_block_and_worker_timeout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        b1 = _make_board(root / "board_a.tif")
        manifest = build_candidate_verifier_manifest(
            source_predictions=[
                {"prediction_id": "p1", "board_id": "b1", "image_id": 1, "orig_order": 0, "label": "ic", "bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "score": 1.0},
                {"prediction_id": "p2", "board_id": "b1", "image_id": 1, "orig_order": 1, "label": "ic", "bbox_xyxy": [220.0, 220.0, 320.0, 320.0], "score": 1.0},
            ],
            board_image_paths={"b1": b1},
            crops_dir=root / "crops",
            inference_spec=_BASE_SPEC,
            source_provenance={"note": "timeout_test"},
        )
        mpath = root / "manifest.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")

        # 1. VRAM gate failed -> blocks immediately
        marker = root / "worker_should_not_run.txt"
        blocked_res = run_verifier_supervised(
            manifest_path=mpath,
            output_dir=root / "out_blocked",
            gpu_gate_fn=lambda a, b: {"passed": False},
            worker_cmd=[sys.executable, "-c", f"open(r'{marker}','w').write('ran')"],
            execute=True,
        )
        assert blocked_res["stage_status"] == "gate_failed"
        assert blocked_res["attempted"] == 0
        assert blocked_res["not_started"] == 2
        assert not marker.exists()

        # 2. Fake sleeping worker hits per-call timeout and is killed cleanly
        fake_worker = (
            "import json, sys, time\n"
            "sys.stdout.write(json.dumps({'event':'loaded','load_seconds':0.01})+'\\n'); sys.stdout.flush()\n"
            "line = sys.stdin.readline(); req = json.loads(line)\n"
            "sys.stdout.write(json.dumps({'event':'call_started','call_id':req['call_id']})+'\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        to_res = run_verifier_supervised(
            manifest_path=mpath,
            output_dir=root / "out_timeout",
            gpu_gate_fn=lambda a, b: {"passed": True},
            worker_cmd=[sys.executable, "-u", "-c", fake_worker],
            load_timeout_s=10.0,
            per_call_timeout_s=0.4,
            execute=True,
        )
        assert to_res["stage_status"] == "call_timeout"
        assert to_res["attempted"] == 1
        assert to_res["completed"] == 0
        assert to_res["not_started"] == 1


def test_invalid_crop_preserved_and_leakage_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        b1 = _make_board(root / "board_a.tif")
        manifest = build_candidate_verifier_manifest(
            source_predictions=[
                {"prediction_id": "bad1", "board_id": "b1", "image_id": 1, "orig_order": 0, "label": "ic", "bbox_xyxy": [100.0, 100.0, 90.0, 200.0], "score": 1.0},
            ],
            board_image_paths={"b1": b1},
            crops_dir=root / "crops",
            inference_spec=_BASE_SPEC,
            source_provenance={"note": "invalid_crop_test"},
        )
        assert manifest["counts"]["total_predictions"] == 1
        assert manifest["counts"]["invalid_crops"] == 1
        assert manifest["counts"]["unique_calls"] == 0
        res = apply_verifier_filter_policy(
            predictions=manifest["predictions"],
            verification_records_by_call_id={},
            policy="B_other",
        )
        assert res["counts"]["total_candidates"] == 1
        assert res["counts"]["invalid_crop_filtered"] == 1

        # Leakage check
        bad_spec = dict(_BASE_SPEC)
        bad_prompt = DEFAULT_VERIFIER_PROMPT + " Note TP vs FP."
        bad_spec["prompt"] = bad_prompt
        bad_spec["prompt_sha256"] = hashlib.sha256(bad_prompt.encode("utf-8")).hexdigest()
        bad_manifest = dict(manifest)
        bad_manifest["inference_spec"] = bad_spec
        with pytest.raises(ValueError, match="Forbidden leakage"):
            verify_verifier_manifest_integrity(bad_manifest)
