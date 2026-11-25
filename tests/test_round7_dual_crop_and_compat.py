"""Round 7 tests: legacy B_other/B_exact regression compatibility + dual-crop (10%+40%) consensus."""

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
    apply_dual_crop_consensus_policy,
    apply_verifier_filter_policy,
    build_candidate_verifier_manifest,
)
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    DemoError,
    LiveDemoService,
    resolve_model_preset_spec,
)
from pcb_vlm.evaluation.gt_free_pipeline import (  # noqa: E402
    export_dual_crop_consensus_gt_free,
    export_verified_predictions_gt_free,
)

_EXPECTED_LEGACY_SUMMARY_KEYS = {
    "total_candidates",
    "retained_count",
    "filtered_count",
    "invalid_crop_filtered",
    "rejected_as_other",
    "cross_target_retained_same_orig_class",
    "cross_target_rejected",
    "exact_same_class_retained",
    "format_failure_filtered",
    "technical_failure_filtered",
    "missing_record_count",
}


def _make_board_png(path: pathlib.Path, w: int = 1200, h: int = 900) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[60 : h - 60, 80 : w - 80] = [165, 42, 42]
    arr[180:290, 220:360] = [210, 180, 50]
    arr[380:460, 480:560] = [50, 180, 210]
    arr[540:630, 700:810] = [70, 200, 110]
    arr[200:290, 820:910] = [190, 190, 190]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_legacy_b_other_and_b_exact_dict_and_summary_contracts() -> None:
    preds = [
        {"prediction_id": "p1", "board_id": "b1", "image_id": 1, "orig_order": 0, "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [1.0, 2.0, 30.0, 40.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u1"},
        {"prediction_id": "p2", "board_id": "b1", "image_id": 1, "orig_order": 1, "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [50.0, 60.0, 90.0, 95.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u2"},
        {"prediction_id": "p3", "board_id": "b1", "image_id": 1, "orig_order": 2, "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0, "crop_valid": True, "unique_call_id": "u3"},
    ]
    rows = [
        {"call_id": "u1", "ok": True, "strict_json_ok": True, "strict_label": "resistor"},
        {"call_id": "u2", "ok": True, "strict_json_ok": True, "strict_label": "other"},
        {"call_id": "u3", "ok": True, "strict_json_ok": True, "strict_label": "resistor"},
    ]
    fake_manifest = {"predictions": preds}

    exp_other = export_verified_predictions_gt_free(manifest=fake_manifest, verifier_rows=rows, policy="B_other")
    assert set(exp_other["summary"].keys()) == _EXPECTED_LEGACY_SUMMARY_KEYS
    assert exp_other["per_board"]["b1"]["pred_record"]["objects"] == [
        {"label": "capacitor", "bbox_xyxy": [1.0, 2.0, 30.0, 40.0], "score": 1.0},
        {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0},
    ]
    for d in exp_other["per_board"]["b1"]["decisions"]:
        assert "technical_anomaly" not in d

    exp_exact = export_verified_predictions_gt_free(manifest=fake_manifest, verifier_rows=rows, policy="B_exact")
    assert set(exp_exact["summary"].keys()) == _EXPECTED_LEGACY_SUMMARY_KEYS
    assert exp_exact["per_board"]["b1"]["pred_record"]["objects"] == [
        {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 130.0, 120.0], "score": 1.0},
    ]


def test_dual_crop_consensus_matrix_and_negative_cases() -> None:
    """Cover dual-other delete, Stage-2 same/cross rescue, and missing/invalid/format/truncated/tech_fail negative cases."""
    s1_preds = [
        {"prediction_id": "p0_s1_keep", "board_id": "b1", "orig_order": 0, "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [10, 10, 40, 40], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u0"},
        {"prediction_id": "p1_dual_other", "board_id": "b1", "orig_order": 1, "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [50, 50, 80, 70], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u1"},
        {"prediction_id": "p2_rescue_exact", "board_id": "b1", "orig_order": 2, "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [90, 90, 120, 110], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u2"},
        {"prediction_id": "p3_rescue_cross", "board_id": "b1", "orig_order": 3, "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [130, 130, 160, 160], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u3"},
        {"prediction_id": "p4_s2_fmt", "board_id": "b1", "orig_order": 4, "orig_pred_label": "inductor", "orig_pred_bbox_xyxy": [170, 170, 200, 200], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u4"},
        {"prediction_id": "p5_s2_trunc", "board_id": "b1", "orig_order": 5, "orig_pred_label": "ic", "orig_pred_bbox_xyxy": [210, 210, 260, 260], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u5"},
    ]
    s1_recs = {
        "s1_u0": {"call_id": "s1_u0", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "ic"},
        "s1_u1": {"call_id": "s1_u1", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u2": {"call_id": "s1_u2", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u3": {"call_id": "s1_u3", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u4": {"call_id": "s1_u4", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u5": {"call_id": "s1_u5", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
    }
    s2_preds_by_pid = {
        "p1_dual_other": {"prediction_id": "p1_dual_other", "board_id": "b1", "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [50, 50, 80, 70], "crop_valid": True, "unique_call_id": "s2_u1"},
        "p2_rescue_exact": {"prediction_id": "p2_rescue_exact", "board_id": "b1", "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [90, 90, 120, 110], "crop_valid": True, "unique_call_id": "s2_u2"},
        "p3_rescue_cross": {"prediction_id": "p3_rescue_cross", "board_id": "b1", "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [130, 130, 160, 160], "crop_valid": True, "unique_call_id": "s2_u3"},
        "p4_s2_fmt": {"prediction_id": "p4_s2_fmt", "board_id": "b1", "orig_pred_label": "inductor", "orig_pred_bbox_xyxy": [170, 170, 200, 200], "crop_valid": True, "unique_call_id": "s2_u4"},
        "p5_s2_trunc": {"prediction_id": "p5_s2_trunc", "board_id": "b1", "orig_pred_label": "ic", "orig_pred_bbox_xyxy": [210, 210, 260, 260], "crop_valid": True, "unique_call_id": "s2_u5"},
    }
    s2_recs = {
        "s2_u1": {"call_id": "s2_u1", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s2_u2": {"call_id": "s2_u2", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "resistor"},
        "s2_u3": {"call_id": "s2_u3", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "resistor"},  # cross-class still keeps orig 'diode'
        "s2_u4": {"call_id": "s2_u4", "ok": True, "has_normal_eos": True, "strict_json_ok": False, "strict_label": None},
        "s2_u5": {"call_id": "s2_u5", "ok": True, "has_normal_eos": False, "strict_json_ok": False, "strict_label": None},
    }
    res_clean = apply_dual_crop_consensus_policy(
        stage1_predictions=s1_preds,
        stage1_records_by_call_id=s1_recs,
        stage2_predictions_by_pid=s2_preds_by_pid,
        stage2_records_by_call_id=s2_recs,
    )
    assert res_clean["technical_complete"] is True
    c = res_clean["counts"]
    assert c["stage1_retained_count"] == 1
    assert c["stage1_rejected_other_count"] == 5
    assert c["rescued_by_stage2_target_count"] == 2
    assert c["rescued_stage2_exact_same"] == 1
    assert c["rescued_stage2_cross_target"] == 1
    assert c["rejected_dual_crop_other_consensus"] == 1
    assert c["retained_count"] == 5
    assert c["stage2_format_failure_kept_anomaly"] == 1
    assert c["stage2_truncated_kept_anomaly"] == 1
    assert c["total_anomalies_kept"] == 2
    kept = res_clean["kept_predictions_by_board"]["b1"]
    assert [k["prediction_id"] for k in kept] == ["p0_s1_keep", "p2_rescue_exact", "p3_rescue_cross", "p4_s2_fmt", "p5_s2_trunc"]
    assert [k["label"] for k in kept] == ["capacitor", "resistor", "diode", "inductor", "ic"]
    assert [k["orig_order"] for k in kept] == [0, 2, 3, 4, 5]
    assert kept[2]["bbox_xyxy"] == [130.0, 130.0, 160.0, 160.0]

    # Negative: missing Stage-2 record, invalid Stage-2 crop, Stage-2 tech failure -> retain + mark technical_complete=False
    s1_neg = s1_preds + [
        {"prediction_id": "p6_s2_missing", "board_id": "b1", "orig_order": 6, "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [270, 270, 300, 300], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u6"},
        {"prediction_id": "p7_s2_invalid", "board_id": "b1", "orig_order": 7, "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [310, 310, 340, 340], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u7"},
        {"prediction_id": "p8_s2_techfail", "board_id": "b1", "orig_order": 8, "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [350, 350, 380, 380], "score": 1.0, "crop_valid": True, "unique_call_id": "s1_u8"},
    ]
    s1_recs_neg = {
        **s1_recs,
        "s1_u6": {"call_id": "s1_u6", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u7": {"call_id": "s1_u7", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
        "s1_u8": {"call_id": "s1_u8", "ok": True, "has_normal_eos": True, "strict_json_ok": True, "strict_label": "other"},
    }
    s2_preds_neg = {
        **s2_preds_by_pid,
        "p6_s2_missing": {"prediction_id": "p6_s2_missing", "board_id": "b1", "orig_pred_label": "capacitor", "orig_pred_bbox_xyxy": [270, 270, 300, 300], "crop_valid": True, "unique_call_id": "s2_u6"},
        "p7_s2_invalid": {"prediction_id": "p7_s2_invalid", "board_id": "b1", "orig_pred_label": "resistor", "orig_pred_bbox_xyxy": [310, 310, 340, 340], "crop_valid": False, "unique_call_id": None},
        "p8_s2_techfail": {"prediction_id": "p8_s2_techfail", "board_id": "b1", "orig_pred_label": "diode", "orig_pred_bbox_xyxy": [350, 350, 380, 380], "crop_valid": True, "unique_call_id": "s2_u8"},
    }
    s2_recs_neg = {
        **s2_recs,
        # s2_u6 intentionally omitted
        "s2_u8": {"call_id": "s2_u8", "ok": False, "strict_json_ok": False, "strict_label": None},
    }
    res_neg = apply_dual_crop_consensus_policy(
        stage1_predictions=s1_neg,
        stage1_records_by_call_id=s1_recs_neg,
        stage2_predictions_by_pid=s2_preds_neg,
        stage2_records_by_call_id=s2_recs_neg,
    )
    assert res_neg["technical_complete"] is False
    cn = res_neg["counts"]
    assert cn["stage2_missing_record_kept_anomaly"] == 1
    assert cn["stage2_invalid_crop_kept_anomaly"] == 1
    assert cn["stage2_technical_failure_kept_anomaly"] == 1
    assert cn["total_anomalies_kept"] == 5
    assert cn["retained_count"] == 8  # only p1_dual_other rejected!


def test_r7_dual_crop_manifest_dedup_and_http_modes(tmp_path: pathlib.Path) -> None:
    """Verify 40% crop padding + dedup, full 2-pass HTTP mode, verified reuse_stage1 HTTP mode, and leakage HTTP 400."""
    for pname in ("r7_dual_crop_consensus", "r7_qwen3_tile768_dual_crop_consensus"):
        spec = resolve_model_preset_spec("qwen3", pname)
        assert spec["preset_id"] == "r7_qwen3_tile768_dual_crop_consensus"
        assert spec["verifier_policy"] == "B_other_dual_crop_consensus"
        assert spec["verifier_pad_fraction"] == 0.10
        assert spec["verifier_stage2_pad_fraction"] == 0.40
        for bad_mk in ("qwen2", "deepseek", "yolo11n"):
            with pytest.raises(DemoError):
                resolve_model_preset_spec(bad_mk, pname)

    img_p = tmp_path / "syn_red.png"
    img_bytes = _make_board_png(img_p, 1200, 900)

    def _mock_dual_verifier(job_spec, manifest_path, cancel_ev, svc_ref):
        jdir = manifest_path.parent
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        pad = float((man.get("crop_spec") or {}).get("pad_fraction") or 0.10)
        out_dir = jdir / "verifier_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, u in enumerate(man["unique_calls"]):
            if abs(pad - 0.10) < 1e-6:
                # idx=0 kept by Stage-1; idx>=1 become 'other' in Stage-1
                lbl = "capacitor" if idx == 0 else "other"
                raw_out = json.dumps({"label": lbl}) + "<|im_end|>"
                s_ok, s_lbl, has_eos = True, lbl, True
            else:
                # Stage-2 (40% pad): idx=0 rescued as resistor; idx=1 stays other (deleted); idx=2 truncated anomaly (kept)
                if idx == 0:
                    raw_out, s_ok, s_lbl, has_eos = '{"label": "resistor"}<|im_end|>', True, "resistor", True
                elif idx == 1:
                    raw_out, s_ok, s_lbl, has_eos = '{"label": "other"}<|im_end|>', True, "other", True
                else:
                    raw_out, s_ok, s_lbl, has_eos = '{"label": "ic', False, None, False
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
                "generated_tokens": 8 if has_eos else 32,
                "has_normal_eos": has_eos,
                "raw_output": raw_out,
                "strict_json_ok": s_ok,
                "strict_label": s_lbl,
                "compat_regex_label": s_lbl,
                "parse_error": None if s_ok else "truncated",
                "gpu_generate_ms": 40.0,
                "cuda_max_reserved_mib": 4100.0,
                "ok": True,
            })
        (out_dir / "verifier_raw_outputs.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws", verifier_runner=_mock_dual_verifier)
    server = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws", service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        st_i, _, html_i = _http_req(f"{base}/")
        assert st_i == 200 and 'value="r7_qwen3_tile768_dual_crop_consensus"' in html_i.decode("utf-8")

        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=img_bytes, headers={"X-Filename": "syn_red.png"})
        assert st_u == 200
        up = json.loads(raw_u.decode("utf-8"))

        # Include duplicate box coordinates for c1_rescue & c1_dup so Stage-2 deduplication is tested too!
        cands = [
            {"prediction_id": "c0", "orig_order": 0, "label": "capacitor", "bbox_xyxy": [220.0, 180.0, 360.0, 290.0], "score": 1.0},
            {"prediction_id": "c1", "orig_order": 1, "label": "diode", "bbox_xyxy": [480.0, 380.0, 560.0, 460.0], "score": 1.0},
            {"prediction_id": "c2", "orig_order": 2, "label": "resistor", "bbox_xyxy": [480.0, 380.0, 560.0, 460.0], "score": 1.0},  # exact crop dedup with c1!
            {"prediction_id": "c3", "orig_order": 3, "label": "inductor", "bbox_xyxy": [700.0, 540.0, 810.0, 630.0], "score": 1.0},
            {"prediction_id": "c4", "orig_order": 4, "label": "ic", "bbox_xyxy": [820.0, 200.0, 910.0, 290.0], "score": 1.0},
        ]

        # 1. Full 2-pass dual-crop HTTP mode (Stage-1 10% -> Stage-2 40%)
        st_s1, _, raw_s1 = _http_req(
            f"{base}/api/jobs/verify_candidates",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "predictions": cands, "preset": "r7_qwen3_tile768_dual_crop_consensus"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_s1 == 202
        jid1 = json.loads(raw_s1.decode("utf-8"))["job_id"]
        for _ in range(100):
            _, _, r_st = _http_req(f"{base}/api/jobs/{jid1}")
            js = json.loads(r_st.decode("utf-8"))
            if js["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.05)
        assert js["status"] == "completed" and js["technical_complete"] is True
        res1 = json.loads(_http_req(f"{base}/api/jobs/{jid1}/result")[2].decode("utf-8"))
        vs1 = res1["verifier_summary"]
        assert vs1["stage1_retained_count"] == 1
        assert vs1["stage1_rejected_other_count"] == 4
        assert vs1["rescued_by_stage2_target_count"] == 2  # c1 and dedup c2 both rescued!
        assert vs1["rejected_dual_crop_other_consensus"] == 1  # c3 deleted!
        assert vs1["stage2_truncated_kept_anomaly"] == 1  # c4 kept as anomaly!
        assert [p["prediction_id"] for p in res1["predictions"]] == ["c0", "c1", "c2", "c4"]
        assert [p["label"] for p in res1["predictions"]] == ["capacitor", "diode", "resistor", "ic"]

        # Verify 40% crop box on c1 is strictly wider than 10% crop box
        s1_man_p = tmp_path / "ws" / "jobs" / jid1 / "verifier_manifest.json"
        s2_man_p = tmp_path / "ws" / "jobs" / jid1 / "stage2" / "verifier_manifest.json"
        s1_man = json.loads(s1_man_p.read_text(encoding="utf-8"))
        s2_man = json.loads(s2_man_p.read_text(encoding="utf-8"))
        assert s1_man["counts"]["unique_calls"] == 4
        assert s2_man["counts"]["unique_calls"] == 3  # c1+c2 deduped to 1!
        assert s1_man["predictions"][1]["crop_box_padded_xyxy"] == [472, 372, 568, 468]
        assert s2_man["predictions"][0]["crop_box_padded_xyxy"] == [448, 348, 592, 492]

        # 2. Leak rejection in reuse_stage1 -> HTTP 400
        st_bad_r, _, _ = _http_req(
            f"{base}/api/jobs/verify_candidates",
            method="POST",
            body=json.dumps({
                "upload_id": up["upload_id"],
                "predictions": cands,
                "preset": "r7_qwen3_tile768_dual_crop_consensus",
                "reuse_stage1": {"manifest_path": str(s1_man_p), "raw_outputs_path": str(tmp_path / "ws" / "jobs" / jid1 / "verifier_runs" / "verifier_raw_outputs.jsonl"), "tp": 1},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_bad_r == 400

        # 3. Valid Stage-1 reuse mode runs Stage-2 ONLY cleanly!
        st_s2, _, raw_s2 = _http_req(
            f"{base}/api/jobs/verify_candidates",
            method="POST",
            body=json.dumps({
                "upload_id": up["upload_id"],
                "predictions": cands,
                "preset": "r7_qwen3_tile768_dual_crop_consensus",
                "reuse_stage1": {
                    "manifest_path": str(s1_man_p),
                    "raw_outputs_path": str(tmp_path / "ws" / "jobs" / jid1 / "verifier_runs" / "verifier_raw_outputs.jsonl"),
                },
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_s2 == 202
        jid2 = json.loads(raw_s2.decode("utf-8"))["job_id"]
        for _ in range(100):
            js2 = json.loads(_http_req(f"{base}/api/jobs/{jid2}")[2].decode("utf-8"))
            if js2["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.05)
        assert js2["status"] == "completed" and js2["technical_complete"] is True
        res2 = json.loads(_http_req(f"{base}/api/jobs/{jid2}/result")[2].decode("utf-8"))
        assert res2["dual_crop_provenance"]["stage1_reused"] is True
        assert res2["dual_crop_provenance"]["stage2_unique_calls"] == 3
        assert len(res2["calls"]) == 3
        assert [p["prediction_id"] for p in res2["predictions"]] == ["c0", "c1", "c2", "c4"]

        st_md, _, md_bytes = _http_req(f"{base}/api/jobs/{jid2}/export/summary_md")
        assert st_md == 200
        md_txt = md_bytes.decode("utf-8")
        assert "外部导入的已保存真实推理候选框" in md_txt
        assert "双裁片一致" in md_txt
    finally:
        svc.shutdown()
        server.shutdown()
        server.server_close()
