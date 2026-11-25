"""CPU unit and HTTP integration tests for Round 14 composite detection system."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (_SRC_ROOT, _SCRIPTS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    YOLO_WACV_ID_TO_RAW_NAME,
    apply_bounded_vlm_verification,
    class_wise_nms_with_trace,
    evaluate_composite_board,
    evaluate_three_way_comparison,
    execute_composite_query,
    normalize_yolo_raw_class,
    parse_composite_query,
    select_hard_candidates_for_vlm,
    split_aligned_and_extra_predictions,
)
from pcb_vlm.evaluation.candidate_verifier import DEFAULT_VERIFIER_PROMPT  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402
from pcb_vlm.evaluation.gt_free_pipeline import prepare_verifier_manifest_gt_free  # noqa: E402
import fics_final_demo  # noqa: E402


def _make_pcb_test_png(path: pathlib.Path, size: tuple[int, int] = (400, 300)) -> None:
    im = Image.new("RGB", size, (18, 92, 54))
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 40, 120, 90], fill=(220, 180, 70))
    draw.rectangle([140, 40, 180, 80], fill=(120, 140, 130))
    draw.rectangle([200, 40, 240, 80], fill=(40, 40, 45))
    draw.rectangle([20, 200, 60, 240], fill=(190, 90, 90))
    draw.rectangle([260, 150, 340, 220], fill=(70, 150, 220))
    im.save(path)


def test_wacv_alignment_never_merges_led_or_extra_classes() -> None:
    led = normalize_yolo_raw_class(4, YOLO_WACV_ID_TO_RAW_NAME)
    assert led["raw_yolo_label"] == "led"
    assert led["aligned_fics6_label"] is None
    assert led["is_fics6_aligned"] is False

    conn = normalize_yolo_raw_class(3, YOLO_WACV_ID_TO_RAW_NAME)
    assert conn["raw_yolo_label"] == "connector"
    assert conn["is_fics6_aligned"] is False

    res = normalize_yolo_raw_class(1, YOLO_WACV_ID_TO_RAW_NAME)
    assert res["aligned_fics6_label"] == "resistor"
    assert res["is_fics6_aligned"] is True


def test_class_wise_nms_and_no_top200_truncation() -> None:
    dets = []
    # Create 230 non-overlapping resistors with scores in [0.05, 0.90]
    for i in range(230):
        x1 = (i % 23) * 30.0
        y1 = (i // 23) * 30.0
        sc = 0.06 + 0.003 * i
        info = normalize_yolo_raw_class(1, YOLO_WACV_ID_TO_RAW_NAME)
        dets.append({
            "pre_nms_id": f"r_{i}",
            "tile_id": 0,
            "tile_box_index": i,
            "score": sc,
            "bbox_xyxy": [x1, y1, x1 + 20.0, y1 + 20.0],
            **info,
        })
    # Add overlapping same-location LED and diode to verify class-wise independence
    dets.append({
        "pre_nms_id": "led_0",
        "tile_id": 0,
        "tile_box_index": 500,
        "score": 0.80,
        "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
        **normalize_yolo_raw_class(4, YOLO_WACV_ID_TO_RAW_NAME),
    })
    dets.append({
        "pre_nms_id": "diode_0",
        "tile_id": 0,
        "tile_box_index": 501,
        "score": 0.70,
        "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
        **normalize_yolo_raw_class(6, YOLO_WACV_ID_TO_RAW_NAME),
    })

    kept = class_wise_nms_with_trace(dets, iou_threshold=0.50, label_key="raw_yolo_label", board_id="b1")
    split = split_aligned_and_extra_predictions(kept, conf_official=0.25)

    assert len(split["low_conf_pool_fics6"]) == 231  # ZERO Top-200 cap!
    assert len(split["official_predictions_fics6"]) > 150
    assert split["non_fics_class_counts_025"] == {"led": 1}
    assert all(p["label"] != "led" for p in split["official_predictions_fics6"])


def test_bounded_candidate_selector_and_conservative_vlm_verification(tmp_path: pathlib.Path) -> None:
    img_p = tmp_path / "board.png"
    _make_pcb_test_png(img_p, (400, 300))

    pool = []
    # Cross-class conflict pair (>= 0.25 overlapping IoU >= 0.50)
    pool.append({
        "prediction_id": "c_conf_1", "board_id": "b1", "image_id": 1, "label": "resistor",
        "score": 0.42, "bbox_xyxy": [40.0, 40.0, 100.0, 80.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    pool.append({
        "prediction_id": "c_conf_2", "board_id": "b1", "image_id": 1, "label": "capacitor",
        "score": 0.38, "bbox_xyxy": [41.0, 40.0, 101.0, 80.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    # Uncertain official boxes 0.25..0.45
    pool.append({
        "prediction_id": "u_rej", "board_id": "b1", "image_id": 1, "label": "diode",
        "score": 0.28, "bbox_xyxy": [140.0, 40.0, 180.0, 80.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    pool.append({
        "prediction_id": "u_fallback", "board_id": "b1", "image_id": 1, "label": "ic",
        "score": 0.31, "bbox_xyxy": [200.0, 40.0, 240.0, 80.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    # Borderline 0.12..0.25 rescue vs reject
    pool.append({
        "prediction_id": "low_rescue", "board_id": "b1", "image_id": 1, "label": "inductor",
        "score": 0.20, "bbox_xyxy": [260.0, 150.0, 320.0, 200.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    pool.append({
        "prediction_id": "low_no_rescue", "board_id": "b1", "image_id": 1, "label": "resistor",
        "score": 0.16, "bbox_xyxy": [20.0, 200.0, 60.0, 240.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })
    # High-confidence unselected YOLO box (should remain untouched)
    pool.append({
        "prediction_id": "strong_unselected", "board_id": "b1", "image_id": 1, "label": "capacitor",
        "score": 0.89, "bbox_xyxy": [320.0, 20.0, 360.0, 60.0], "detector_source": "yolo11n", "vlm_intervened": False,
    })

    sel = select_hard_candidates_for_vlm(pool, max_candidates=48)
    selected = sel["selected_candidates"]
    sel_ids = [s["prediction_id"] for s in selected]
    assert "strong_unselected" not in sel_ids
    assert set(["c_conf_1", "c_conf_2", "u_rej", "u_fallback", "low_rescue", "low_no_rescue"]).issubset(sel_ids)

    spec = {
        "model_key": "qwen3", "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct", "model_revision": "rev1",
        "model_local_dir": "local", "overlay_dir": "ov", "dtype": "bfloat16", "do_sample": False,
        "repetition_penalty": 1.0, "max_new_tokens": 32, "min_pixels": 65536, "max_pixels": 200704,
        "allowed_labels": ["capacitor", "resistor", "inductor", "transistor", "diode", "ic", "other"],
        "prompt": DEFAULT_VERIFIER_PROMPT,
        "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
    }
    manifest, _ = prepare_verifier_manifest_gt_free(
        reconstructed_by_board=[{"board_id": "b1", "image_id": 1, "mode": "COMPOSITE_YOLO_QWEN3", "technical_complete": True, "predictions": selected}],
        board_image_paths={"b1": img_p},
        crops_dir=tmp_path / "crops",
        inference_spec=spec,
        source_provenance={"board_id": "b1"},
        pad_fraction=0.25,
    )
    uid_by_pid = {p["prediction_id"]: p["unique_call_id"] for p in manifest["predictions"]}
    # Use exact runner schema ("raw_output", "has_normal_eos", "ok")
    verifier_rows = [
        {"call_id": uid_by_pid["c_conf_1"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "resistor"}<|im_end|>'},
        {"call_id": uid_by_pid["c_conf_2"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "resistor"}<|im_end|>'},
        {"call_id": uid_by_pid["u_rej"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "other"}<|im_end|>'},
        {"call_id": uid_by_pid["u_fallback"], "ok": True, "has_normal_eos": False, "raw_output": '{"label": "other"'},  # non-EOS must conservatively keep YOLO!
        {"call_id": uid_by_pid["low_rescue"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "inductor"}<|im_end|>'},
        {"call_id": uid_by_pid["low_no_rescue"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "capacitor"}<|im_end|>'},
    ]
    out = apply_bounded_vlm_verification(
        board_id="b1",
        low_conf_pool_fics6=pool,
        selected_candidates=selected,
        manifest=manifest,
        verifier_rows=verifier_rows,
    )
    dc = out["decision_counts"]
    assert dc["rejected_background"] == 1  # u_rej
    assert dc["vlm_fallback_keep_yolo"] == 1  # u_fallback
    assert dc["rescued_low_conf_agree"] == 1  # low_rescue
    assert dc["borderline_not_rescued_conservative"] == 1  # low_no_rescue
    assert dc["unselected_yolo_kept"] == 1  # strong_unselected

    final_by_pid = {p["prediction_id"]: p for p in out["final_predictions"]}
    assert "u_rej" not in final_by_pid
    assert final_by_pid["u_fallback"]["label"] == "ic"
    assert final_by_pid["u_fallback"]["vlm_intervened"] is False
    assert final_by_pid["low_rescue"]["detector_source"] == "yolo11n+qwen3"
    assert final_by_pid["low_rescue"]["vlm_intervened"] is True
    assert final_by_pid["strong_unselected"]["detector_source"] == "yolo11n"
    # c_conf_1 & c_conf_2 were both relabeled to resistor at same location -> final class-wise NMS deduplicates to 1
    assert "c_conf_1" in final_by_pid and "c_conf_2" not in final_by_pid


def test_composite_query_parsing_and_execution() -> None:
    p1 = parse_composite_query("找电阻，但只统计左半边")
    assert p1["ok"] and p1["target_classes"] == ["resistor"] and p1["region_mode"] == "left_half"
    p_bad = parse_composite_query("只找 LED 灯")
    assert not p_bad["ok"] and p_bad["error_code"] == "out_of_scope_class"

    preds = [
        {"prediction_id": "p1", "label": "resistor", "score": 0.9, "bbox_xyxy": [20.0, 40.0, 60.0, 80.0], "detector_source": "yolo11n", "vlm_intervened": False},
        {"prediction_id": "p2", "label": "resistor", "score": 0.4, "bbox_xyxy": [250.0, 40.0, 310.0, 80.0], "detector_source": "yolo11n+qwen3", "vlm_intervened": True},
        {"prediction_id": "p3", "label": "capacitor", "score": 0.8, "bbox_xyxy": [30.0, 100.0, 70.0, 140.0], "detector_source": "yolo11n", "vlm_intervened": False},
    ]
    q_left = execute_composite_query(
        predictions=preds,
        query_text="只找电阻，但只统计左半边",
        orig_wh=(400, 300),
        base_job_mode="COMPOSITE_YOLO_QWEN3",
        vlm_calls_count=2,
    )
    assert q_left["ok"]
    assert q_left["execution_path"] == "composite_with_qwen3_bounded_verify"
    assert q_left["gpu_called_for_query"] is False
    assert q_left["predicted_count"] == 1
    assert q_left["class_matched_excluded_by_region"] == 1
    assert q_left["coordinate_convention"]["split_value_px"] == 200.0

    q_absent = execute_composite_query(
        predictions=preds,
        query_text="只找电感并计数",
        orig_wh=(400, 300),
        base_job_mode="COMPOSITE_YOLO",
    )
    assert q_absent["ok"]
    assert q_absent["predicted_count"] == 0
    assert q_absent["execution_path"] == "composite_existing_result_filter"


def test_composite_evaluator_and_promotion_gate() -> None:
    gts = [
        {"id": 1, "label": "resistor", "bbox_xyxy": [10.0, 10.0, 50.0, 30.0]},
        {"id": 2, "label": "capacitor", "bbox_xyxy": [100.0, 10.0, 140.0, 30.0]},
        {"id": 3, "label": "diode", "bbox_xyxy": [200.0, 10.0, 240.0, 30.0]},
        {"id": 4, "label": "inductor", "bbox_xyxy": [300.0, 10.0, 340.0, 30.0]},
    ]
    preds_025 = [
        {"prediction_id": "a", "label": "resistor", "score": 0.8, "bbox_xyxy": [10.0, 10.0, 50.0, 30.0]},
        {"prediction_id": "b", "label": "resistor", "score": 0.6, "bbox_xyxy": [100.0, 10.0, 140.0, 30.0]},  # wrong class for GT 2!
    ]
    pool_005 = preds_025 + [
        {"prediction_id": "c", "label": "diode", "score": 0.18, "bbox_xyxy": [200.0, 10.0, 240.0, 30.0]},  # filtered <0.25!
    ]
    ev = evaluate_composite_board(gts=gts, predictions_025=preds_025, low_conf_pool_005=pool_005)
    assert ev["class_aware"]["tp"] == 1
    assert ev["class_agnostic_coverage"]["pool_005_iou50"]["matched_gt"] == 3
    ed = ev["error_decomposition"]
    assert ed["fn_wrong_class_at_iou50"] == 1
    assert ed["fn_score_filtered_below_025_present_in_005"] == 1
    assert ed["fn_true_localization_miss"] == 1

    cmp_eval = evaluate_three_way_comparison(per_board_results={
        "s20": {
            "qwen3_r5": {"class_aware": {"tp": 33, "fp": 278, "fn": 78}},
            "yolo11n_direct": {"class_aware": {"tp": 40, "fp": 20, "fn": 71}},
            "yolo11n_qwen3_composite": {"class_aware": {"tp": 45, "fp": 18, "fn": 66}},
        },
        "s4": {
            "qwen3_r5": {"class_aware": {"tp": 32, "fp": 251, "fn": 20}},
            "yolo11n_direct": {"class_aware": {"tp": 30, "fp": 15, "fn": 22}},
            "yolo11n_qwen3_composite": {"class_aware": {"tp": 33, "fp": 14, "fn": 19}},
        },
    })
    assert cmp_eval["promotion_gate"]["passed"] is True


def test_http_composite_service_end_to_end_and_pure_vlm_isolation(tmp_path: pathlib.Path) -> None:
    ws = tmp_path / "ws"
    img_p = tmp_path / "sample_pcb.png"
    _make_pcb_test_png(img_p, (400, 300))

    def fake_yolo(job_spec, plan, comp_out, cancel_ev, service):
        res_info = normalize_yolo_raw_class(1, YOLO_WACV_ID_TO_RAW_NAME)
        ind_info = normalize_yolo_raw_class(7, YOLO_WACV_ID_TO_RAW_NAME)
        led_info = normalize_yolo_raw_class(4, YOLO_WACV_ID_TO_RAW_NAME)
        q_items = plan["queues"]["yolo11n"]
        raw_records = [
            {
                "call_id": q["call_id"], "call_index": q["call_index"], "board_id": job_spec["board_id"],
                "tile_id": q["tile_id"], "width": q["width"], "height": q["height"],
                "parsed_count": 1, "raw_output": "[]", "gpu_generate_ms": 5.0,
            }
            for q in q_items
        ]
        pool = [
            {
                "prediction_id": "p_res_left", "board_id": job_spec["board_id"], "image_id": 1,
                "label": "resistor", "score": 0.85, "bbox_xyxy": [40.0, 40.0, 120.0, 90.0],
                "detector_source": "yolo11n", "vlm_intervened": False, **res_info,
            },
            {
                "prediction_id": "p_res_right", "board_id": job_spec["board_id"], "image_id": 1,
                "label": "resistor", "score": 0.21, "bbox_xyxy": [260.0, 150.0, 340.0, 220.0],
                "detector_source": "yolo11n", "vlm_intervened": False, **res_info,
            },
        ]
        official = [pool[0]]
        return {
            "summary": {"counts": {"pre_nms_all": 3}, "timing_and_vram": {"cuda_max_memory_reserved_mib": 120.0}},
            "low_conf_pool_fics6": pool,
            "official_predictions_fics6": official,
            "extra_info": {"counts_005": {led_info["raw_yolo_label"]: 1}, "counts_025": {led_info["raw_yolo_label"]: 1}},
            "raw_records": raw_records,
        }

    def fake_verifier(job_spec, manifest_path, cancel_ev, service):
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        out_dir = manifest_path.parent / "verifier_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"call_id": c["call_id"], "ok": True, "has_normal_eos": True, "raw_output": '{"label": "resistor"}<|im_end|>'}
            for c in man["unique_calls"]
        ]
        (out_dir / "verifier_raw_outputs.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    srv = fics_final_demo.create_demo_server(
        workspace_root=ws,
        host="127.0.0.1",
        port=0,
        service=LiveDemoService(
            workspace_root=ws,
            verifier_runner=fake_verifier,
            composite_yolo_runner=fake_yolo,
        ),
    )
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        req = urllib.request.Request(
            f"{base}/api/upload",
            data=img_p.read_bytes(),
            headers={"X-Filename": "sample_pcb.png"},
            method="POST",
        )
        up = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))

        # Check Entry A / Entry B defaults intact
        cfg = json.loads(urllib.request.urlopen(f"{base}/api/config", timeout=10).read().decode("utf-8"))
        assert cfg["single_model_defense"]["model_key"] == "qwen3"
        assert cfg["single_model_defense"]["mode"] == "T"
        assert cfg["three_model_compare"]["mode"] == "W"

        # Submit COMPOSITE_YOLO_QWEN3 job
        j_req = urllib.request.Request(
            f"{base}/api/jobs",
            data=json.dumps({"upload_id": up["upload_id"], "model_key": "yolo11n_qwen3_composite", "mode": "COMPOSITE_YOLO_QWEN3"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        job = json.loads(urllib.request.urlopen(j_req, timeout=10).read().decode("utf-8"))
        jid = job["job_id"]
        for _ in range(50):
            st = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}", timeout=10).read().decode("utf-8"))
            if st["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.05)
        assert st["status"] == "completed"
        res = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}/result", timeout=10).read().decode("utf-8"))
        assert res["gt_loaded"] is False
        assert res["kept_predictions_count"] == 2  # p_res_left + rescued p_res_right!
        assert res["composite_metadata"]["raw_non_fics_class_counts_025"] == {"led": 1}

        # Run compound spatial query
        q_req = urllib.request.Request(
            f"{base}/api/composite/query",
            data=json.dumps({"job_id": jid, "query_text": "只找电阻，但只统计左半边"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        q_out = json.loads(urllib.request.urlopen(q_req, timeout=10).read().decode("utf-8"))
        assert q_out["predicted_count"] == 1
        assert q_out["class_matched_excluded_by_region"] == 1
        assert q_out["execution_path"] == "composite_with_qwen3_bounded_verify"
        ov_png = urllib.request.urlopen(f"{base}{q_out['overlay_url']}", timeout=10).read()
        assert ov_png.startswith(b"\x89PNG")

        # Post-hoc COCO GT eval scoped to left half
        coco = {
            "images": [{"id": 1, "file_name": "sample_pcb.png", "width": 400, "height": 300}],
            "categories": [
                {"id": 1, "name": "capacitor"}, {"id": 2, "name": "resistor"}, {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"}, {"id": 5, "name": "diode"}, {"id": 6, "name": "ic"},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 2, "bbox": [40.0, 40.0, 80.0, 50.0], "area": 4000.0, "iscrowd": 0},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [260.0, 150.0, 80.0, 70.0], "area": 5600.0, "iscrowd": 0},
            ],
        }
        ev_req = urllib.request.Request(
            f"{base}/api/composite/query_evaluate_gt",
            data=json.dumps({"job_id": jid, "query_id": q_out["query_id"], "coco_gt": coco}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ev_out = json.loads(urllib.request.urlopen(ev_req, timeout=10).read().decode("utf-8"))
        assert ev_out["post_hoc_query_eval"]["gt_count"] == 1
        assert ev_out["post_hoc_query_eval"]["tp"] == 1
        assert ev_out["post_hoc_query_eval"]["f1"] == 1.0
    finally:
        srv.shutdown()
        srv.demo_service.shutdown()
        srv.server_close()
