"""CPU, strict B_other_strict_keep_anomaly, and HTTP Stage-2 verifier tests for Round 6."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fics_final_demo  # noqa: E402
from pcb_vlm.evaluation.candidate_verifier import (  # noqa: E402
    DEFAULT_VERIFIER_PROMPT,
    apply_verifier_filter_policy,
)
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    DemoError,
    LiveDemoService,
    resolve_model_preset_spec,
)
from pcb_vlm.evaluation.gt_free_pipeline import (  # noqa: E402
    export_verified_predictions_gt_free,
)


def _make_board_png(path: pathlib.Path, w: int = 1200, h: int = 900) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[80 : h - 80, 100 : w - 100] = [40, 115, 68]
    arr[200:340, 260:420] = [210, 180, 50]
    arr[500:640, 620:780] = [50, 180, 210]
    arr[620:720, 820:940] = [190, 90, 90]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_b_other_strict_keep_anomaly_delete_only_other_and_count_anomalies() -> None:
    """Verify strict B_other deletes ONLY strict 'other', keeps & counts format/truncated/failed/missing anomalies, and marks incomplete when missing/failed."""
    preds = [
        {"prediction_id": "p0", "board_id": "b1", "orig_order": 0, "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "crop_valid": True, "unique_call_id": "u0"},
        {"prediction_id": "p1", "board_id": "b1", "orig_order": 1, "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [60.0, 60.0, 100.0, 100.0], "crop_valid": True, "unique_call_id": "u1"},
        {"prediction_id": "p2", "board_id": "b1", "orig_order": 2, "orig_pred_label": "ic", "orig_pred_bbox_xyxy": [110.0, 110.0, 180.0, 180.0], "crop_valid": True, "unique_call_id": "u2"},
        {"prediction_id": "p3", "board_id": "b1", "orig_order": 3, "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [200.0, 200.0, 240.0, 240.0], "crop_valid": True, "unique_call_id": "u3"},
        {"prediction_id": "p4", "board_id": "b1", "orig_order": 4, "orig_pred_label": "inductor", "orig_pred_bbox_xyxy": [250.0, 250.0, 300.0, 300.0], "crop_valid": True, "unique_call_id": "u4"},
    ]
    records_all_ok_with_fmt = {
        "u0": {"call_id": "u0", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "capacitor"},
        "u1": {"call_id": "u1", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "capacitor"},  # cross-target keep orig 'resistor'
        "u2": {"call_id": "u2", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},      # delete ONLY this!
        "u3": {"call_id": "u3", "ok": True, "has_normal_eos": False, "strict_json_ok": False, "strict_label": None},       # truncated -> keep orig!
        "u4": {"call_id": "u4", "ok": True, "has_normal_eos": True, "strict_json_ok": False, "strict_label": None},        # format failure -> keep orig!
    }
    res1 = apply_verifier_filter_policy(
        predictions=preds,
        verification_records_by_call_id=records_all_ok_with_fmt,
        policy="B_other_strict_keep_anomaly",
    )
    assert res1["technical_complete"] is True
    c1 = res1["counts"]
    assert c1["total_candidates"] == 5
    assert c1["rejected_as_other"] == 1
    assert c1["retained_count"] == 4
    assert c1["exact_same_class_retained"] == 1
    assert c1["cross_target_retained_same_orig_class"] == 1
    assert c1["truncated_kept_anomaly"] == 1
    assert c1["format_failure_kept_anomaly"] == 1
    assert c1["total_anomalies_kept"] == 2
    kept1 = res1["kept_predictions_by_board"]["b1"]
    assert [k["prediction_id"] for k in kept1] == ["p0", "p1", "p3", "p4"]
    assert [k["label"] for k in kept1] == ["capacitor", "resistor", "diode", "inductor"]
    assert kept1[1]["bbox_xyxy"] == [60.0, 60.0, 100.0, 100.0]
    assert kept1[2]["technical_anomaly"] is True and kept1[2]["anomaly_reason"] == "truncated_kept_anomaly"
    assert kept1[3]["technical_anomaly"] is True and kept1[3]["anomaly_reason"] == "format_failure_kept_anomaly"

    # Any missing or ok=False call MUST mark technical_complete=False while still retaining box + counting separately
    preds_with_invalid = preds + [
        {"prediction_id": "p5", "board_id": "b1", "orig_order": 5, "orig_pred_label": "transistor", "orig_pred_bbox_xyxy": [1.0, 1.0, 10.0, 10.0], "crop_valid": False, "unique_call_id": None},
    ]
    records_missing_failed = {
        "u0": records_all_ok_with_fmt["u0"],
        "u1": records_all_ok_with_fmt["u1"],
        "u2": records_all_ok_with_fmt["u2"],
        "u3": {"call_id": "u3", "ok": False, "strict_json_ok": False, "strict_label": None},
        # u4 omitted (missing_record)
    }
    res2 = apply_verifier_filter_policy(
        predictions=preds_with_invalid,
        verification_records_by_call_id=records_missing_failed,
        policy="B_other_strict_keep_anomaly",
    )
    assert res2["technical_complete"] is False
    c2 = res2["counts"]
    assert c2["rejected_as_other"] == 1
    assert c2["retained_count"] == 5
    assert c2["technical_failure_kept_anomaly"] == 1
    assert c2["missing_record_kept_anomaly"] == 1
    assert c2["invalid_crop_kept_anomaly"] == 1
    assert c2["total_anomalies_kept"] == 3


def test_r6_preset_restrictions_and_verify_candidates_http_flow(tmp_path: pathlib.Path) -> None:
    """Verify R6 preset restriction, GT-leakage HTTP 400 rejection, isolated Stage-2 verification, and post-hoc GT."""
    for pname in ("r6_b_other_strict", "r6_qwen3_tile768_b_other_strict"):
        spec = resolve_model_preset_spec("qwen3", pname)
        assert spec["preset_id"] == "r6_qwen3_tile768_b_other_strict"
        assert spec["verifier_policy"] == "B_other_strict_keep_anomaly"
        assert spec["explicit_do_sample_false"] is True
        assert spec["tile_size"] == 768
        assert spec["qwen_max_pixels"] == 589824
        for bad_mk in ("qwen2", "deepseek", "yolo11n"):
            with pytest.raises(DemoError):
                resolve_model_preset_spec(bad_mk, pname)

    img_bytes = _make_board_png(tmp_path / "syn_s20_like.png", 1200, 900)

    def _mock_verifier(job_spec, manifest_path, cancel_ev, svc_ref):
        jdir = manifest_path.parent
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        out_dir = jdir / "verifier_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, u in enumerate(man["unique_calls"]):
            if idx == 1:
                raw_out = '{"label": "other"}<|im_end|>'
                strict_ok, strict_lbl, has_eos = True, "other", True
            elif idx == 2:
                raw_out = '{"label": "resis'  # truncated anomaly -> kept!
                strict_ok, strict_lbl, has_eos = False, None, False
            else:
                raw_out = '{"label": "capacitor"}<|im_end|>'
                strict_ok, strict_lbl, has_eos = True, "capacitor", True
            rows.append({
                "call_id": u["call_id"],
                "sample_index": u["sample_index"],
                "neutral_filename": u["neutral_filename"],
                "width": u["width"],
                "height": u["height"],
                "input_sha256": u["image_sha256"],
                "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
                "planned_qwen_min_pixels": 65536,
                "planned_qwen_max_pixels": 200704,
                "processed_width_px": 224,
                "processed_height_px": 224,
                "processed_area_px": 50176,
                "merged_visual_tokens": 64,
                "max_new_tokens": 32,
                "effective_generation_probe": {"effective_generation_mode": "GenerationMode.GREEDY_SEARCH", "effective_do_sample": False},
                "generated_tokens": 9 if has_eos else 32,
                "has_normal_eos": has_eos,
                "raw_output": raw_out,
                "strict_json_ok": strict_ok,
                "strict_label": strict_lbl,
                "compat_regex_label": strict_lbl,
                "parse_error": None if strict_ok else "truncated",
                "gpu_generate_ms": 55.0,
                "cuda_max_reserved_mib": 4100.0,
                "ok": True,
            })
        (out_dir / "verifier_raw_outputs.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws", verifier_runner=_mock_verifier)
    server = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws", service=svc, host="127.0.0.1", port=0)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        st_idx, _, html_idx = _http_req(f"{base}/")
        assert st_idx == 200 and 'value="r6_qwen3_tile768_b_other_strict"' in html_idx.decode("utf-8")

        st, _, raw = _http_req(f"{base}/api/upload", method="POST", body=img_bytes, headers={"X-Filename": "syn_s20_like.png"})
        assert st == 200
        up = json.loads(raw.decode("utf-8"))

        # Reject GT/TP/FP leakage fields with 400
        leak_payload = json.dumps({
            "upload_id": up["upload_id"],
            "predictions": [{"prediction_id": "p0", "label": "capacitor", "bbox_xyxy": [260, 200, 420, 340], "is_tp": True}],
        }).encode("utf-8")
        st_leak, _, _ = _http_req(f"{base}/api/jobs/verify_candidates", method="POST", body=leak_payload, headers={"Content-Type": "application/json"})
        assert st_leak == 400

        # Valid neutral Stage-2 verify request
        neutral_preds = [
            {"prediction_id": "cand_0", "orig_order": 0, "label": "capacitor", "bbox_xyxy": [260.0, 200.0, 420.0, 340.0], "score": 1.0},
            {"prediction_id": "cand_1", "orig_order": 1, "label": "resistor", "bbox_xyxy": [620.0, 500.0, 780.0, 640.0], "score": 1.0},
            {"prediction_id": "cand_2", "orig_order": 2, "label": "ic", "bbox_xyxy": [820.0, 620.0, 940.0, 720.0], "score": 1.0},
        ]
        ok_payload = json.dumps({
            "upload_id": up["upload_id"],
            "predictions": neutral_preds,
            "preset": "r6_qwen3_tile768_b_other_strict",
        }).encode("utf-8")
        st_sub, _, raw_sub = _http_req(f"{base}/api/jobs/verify_candidates", method="POST", body=ok_payload, headers={"Content-Type": "application/json"})
        assert st_sub == 202
        job = json.loads(raw_sub.decode("utf-8"))
        jid = job["job_id"]

        for _ in range(100):
            _, _, r_st = _http_req(f"{base}/api/jobs/{jid}")
            js = json.loads(r_st.decode("utf-8"))
            if js["status"] in ("completed", "failed", "incomplete", "cancelled"):
                break
            time.sleep(0.05)
        assert js["status"] == "completed"
        assert js["technical_complete"] is True

        st_res, _, r_res = _http_req(f"{base}/api/jobs/{jid}/result")
        assert st_res == 200
        res = json.loads(r_res.decode("utf-8"))
        assert res["verifier_only"] is True
        assert res["verifier_policy"] == "B_other_strict_keep_anomaly"
        assert res["verifier_summary"]["rejected_as_other"] == 1
        assert res["verifier_summary"]["truncated_kept_anomaly"] == 1
        assert res["kept_predictions_count"] == 2
        assert [p["prediction_id"] for p in res["predictions"]] == ["cand_0", "cand_2"]
        assert res["predictions"][0]["bbox_xyxy"] == [260.0, 200.0, 420.0, 340.0]
        assert res["predictions"][1]["label"] == "ic"
        assert res["predictions"][1]["technical_anomaly"] is True

        st_md, _, r_md = _http_req(f"{base}/api/jobs/{jid}/export/summary_md")
        assert st_md == 200
        assert b"B_other_strict_keep_anomaly" in r_md

        # Post-hoc COCO GT evaluation works cleanly on Stage-2 verifier job
        coco_gt = {
            "images": [{"id": 1, "file_name": "syn_s20_like.png", "width": 1200, "height": 900, "sha256": up["source_image_sha256"]}],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [260, 200, 160, 140], "area": 22400, "iscrowd": 0},
            ],
        }
        st_gt, _, r_gt = _http_req(
            f"{base}/api/jobs/{jid}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": coco_gt, "target_image_id": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_gt == 200
        ev = json.loads(r_gt.decode("utf-8"))["post_hoc_eval"]["metrics"]
        assert ev["tp"] == 1
        assert ev["fp"] == 1
    finally:
        svc.shutdown()
        server.shutdown()
        server.server_close()
