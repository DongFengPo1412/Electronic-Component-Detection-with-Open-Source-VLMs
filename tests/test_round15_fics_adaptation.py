"""CPU unit and HTTP integration tests for Round 15 FICS 6-class adaptation & preset switching."""

from __future__ import annotations

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

from pcb_vlm.data.fics_dslr import FICS_RAW_TO_CANONICAL  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    YOLO_FICS6_ID_TO_RAW_NAME,
    YOLO_WACV_ID_TO_RAW_NAME,
    evaluate_composite_board,
    execute_composite_query,
    normalize_yolo_raw_class,
    run_yolo11n_class_aware_board,
)
from pcb_vlm.evaluation.final_demo_service import DemoError, LiveDemoService  # noqa: E402
import fics_final_demo  # noqa: E402


def _make_test_board_png(path: pathlib.Path, size: tuple[int, int] = (400, 300)) -> None:
    im = Image.new("RGB", size, (18, 92, 54))
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 40, 100, 80], fill=(220, 180, 70))
    draw.rectangle([250, 100, 360, 220], fill=(70, 150, 220))
    im.save(path)


def test_fics6_schema_and_case_sensitive_ics_lookup() -> None:
    # Ensure exact class order for 6-class adapted weights
    assert YOLO_FICS6_ID_TO_RAW_NAME == {
        0: "capacitor",
        1: "resistor",
        2: "inductor",
        3: "transistor",
        4: "diode",
        5: "ic",
    }
    ic_info = normalize_yolo_raw_class(5, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
    assert ic_info["raw_yolo_label"] == "ic"
    assert ic_info["aligned_fics6_label"] == "ic"
    assert ic_info["is_fics6_aligned"] is True

    # Ensure raw FICS CSV header value "ICs" is preserved accurately
    raw_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    assert FICS_RAW_TO_CANONICAL["ICs"] == "ic"
    assert raw_lower["ics"] == "ic"


def test_evaluate_composite_board_size_bins_and_roi_audit() -> None:
    gts = [
        {"id": 1, "label": "resistor", "bbox_xyxy": [10.0, 10.0, 40.0, 30.0]},       # 600 px^2 -> small (TP)
        {"id": 2, "label": "capacitor", "bbox_xyxy": [100.0, 100.0, 200.0, 180.0]}, # 8000 px^2 -> medium (FN)
        {"id": 3, "label": "ic", "bbox_xyxy": [300.0, 300.0, 500.0, 500.0]},        # 40000 px^2 -> large (TP)
        {"id": 4, "label": "diode", "bbox_xyxy": [800.0, 800.0, 840.0, 840.0]},     # outside ROI [0..700, 0..700]
    ]
    preds_025 = [
        {"prediction_id": "p1", "label": "resistor", "score": 0.9, "bbox_xyxy": [10.0, 10.0, 40.0, 30.0]},
        {"prediction_id": "p2", "label": "ic", "score": 0.8, "bbox_xyxy": [300.0, 300.0, 500.0, 500.0]},
    ]
    pool_005 = preds_025 + [
        {"prediction_id": "p3", "label": "capacitor", "score": 0.15, "bbox_xyxy": [100.0, 100.0, 200.0, 180.0]},
    ]
    ev = evaluate_composite_board(
        gts=gts,
        predictions_025=preds_025,
        low_conf_pool_005=pool_005,
        roi_bbox_half_open=[0, 0, 700, 700],
    )
    sb = ev["per_size_bin"]
    assert sb["small"]["gt_count"] == 2 and sb["small"]["tp"] == 1 and sb["small"]["fn"] == 1
    assert sb["medium"]["gt_count"] == 1 and sb["medium"]["tp"] == 0 and sb["medium"]["ca_005_iou50_any_pred_covered"] == 1
    assert sb["large"]["gt_count"] == 1 and sb["large"]["tp"] == 1
    roi_aud = ev["error_decomposition"]["roi_gt_audit"]
    assert roi_aud["gt_outside_roi"] == 1
    assert roi_aud["gt_partially_clipped_by_roi"] == 0


def test_execute_composite_query_explicit_mechanism_and_participation() -> None:
    preds = [
        {
            "prediction_id": "p1",
            "label": "resistor",
            "score": 0.88,
            "bbox_xyxy": [20.0, 20.0, 60.0, 40.0],
            "detector_source": "yolo11n_fics_r15",
            "vlm_intervened": False,
        }
    ]
    q = execute_composite_query(
        predictions=preds,
        query_text="只找电阻并计数",
        orig_wh=(400, 300),
        base_job_mode="COMPOSITE_YOLO",
        detector_preset="fics_yolo11n_r15",
        yolo_weights_sha256="deadbeef1234",
    )
    assert q["ok"]
    assert q["query_execution_mechanism"] == "筛选已生成的检测结果"
    assert q["detector_preset"] == "fics_yolo11n_r15"
    assert q["yolo_weights_sha256"] == "deadbeef1234"
    mp = q["model_participation"]
    assert mp["stage1_detector_preset"] == "fics_yolo11n_r15"
    assert mp["stage2_vlm_bounded_verify"] is False
    assert mp["query_stage_gpu_called"] is False


def test_http_composite_detector_preset_switching_and_isolation(tmp_path: pathlib.Path) -> None:
    ws = tmp_path / "ws"
    img_p = tmp_path / "board_a.png"
    _make_test_board_png(img_p, (400, 300))

    fake_fics_pt = tmp_path / "fics_r15_best.pt"
    fake_fics_pt.write_bytes(b"mock-fics-pt")
    fake_wacv_pt = tmp_path / "wacv_best.pt"
    fake_wacv_pt.write_bytes(b"mock-wacv-pt")

    seen_calls = []

    def fake_yolo(job_spec, plan, comp_out, cancel_ev, service):
        yolo_m = plan["models"]["yolo11n"]
        dp = yolo_m["detector_preset"]
        schema = yolo_m["expected_class_schema"]
        seen_calls.append((dp, schema, yolo_m["local_dir"]))
        q_items = plan["queues"]["yolo11n"]
        raw_records = [
            {
                "call_id": q["call_id"],
                "call_index": q["call_index"],
                "board_id": job_spec["board_id"],
                "tile_id": q["tile_id"],
                "width": q["width"],
                "height": q["height"],
                "parsed_count": 1,
                "raw_output": "[]",
                "gpu_generate_ms": 4.0,
            }
            for q in q_items
        ]
        if dp == "fics_yolo11n_r15":
            info = normalize_yolo_raw_class(1, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
            preds = [
                {
                    "prediction_id": "fics_p1", "board_id": job_spec["board_id"], "image_id": 1,
                    "label": "resistor", "score": 0.91, "bbox_xyxy": [40.0, 40.0, 100.0, 80.0],
                    "detector_source": "yolo11n_fics_r15", "detector_preset": dp, "vlm_intervened": False, **info,
                },
                {
                    "prediction_id": "fics_p2", "board_id": job_spec["board_id"], "image_id": 1,
                    "label": "ic", "score": 0.87, "bbox_xyxy": [250.0, 100.0, 360.0, 220.0],
                    "detector_source": "yolo11n_fics_r15", "detector_preset": dp, "vlm_intervened": False,
                    **normalize_yolo_raw_class(5, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6"),
                },
            ]
        else:
            info = normalize_yolo_raw_class(1, YOLO_WACV_ID_TO_RAW_NAME, expected_schema="wacv_11")
            preds = [
                {
                    "prediction_id": "wacv_p1", "board_id": job_spec["board_id"], "image_id": 1,
                    "label": "resistor", "score": 0.62, "bbox_xyxy": [40.0, 40.0, 100.0, 80.0],
                    "detector_source": "yolo11n", "detector_preset": dp, "vlm_intervened": False, **info,
                },
            ]
        return {
            "summary": {
                "detector_preset": dp,
                "expected_class_schema": schema,
                "weights_sha256": yolo_m["weights_sha256"],
                "counts": {"pre_nms_all": len(preds)},
                "timing_and_vram": {"cuda_max_memory_reserved_mib": 100.0},
            },
            "low_conf_pool_fics6": preds,
            "official_predictions_fics6": preds,
            "extra_info": {"counts_005": {}, "counts_025": {}},
            "raw_records": raw_records,
        }

    import yaml

    rt_cfg_path = tmp_path / "runtime_config.yaml"
    rt_cfg = {
        "models": {
            "yolo11n": {"local_dir": str(fake_wacv_pt), "weights_sha256": "aaa111"},
            "yolo11n_fics_r15": {"local_dir": str(fake_fics_pt), "weights_sha256": "bbb222"},
        }
    }
    rt_cfg_path.write_text(yaml.safe_dump(rt_cfg), encoding="utf-8")
    svc = LiveDemoService(
        workspace_root=ws,
        runtime_config_path=rt_cfg_path,
        composite_yolo_runner=fake_yolo,
    )
    srv = fics_final_demo.create_demo_server(workspace_root=ws, host="127.0.0.1", port=0, service=svc)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        cfg = json.loads(urllib.request.urlopen(f"{base}/api/config", timeout=10).read().decode("utf-8"))
        presets_map = {p["preset_id"]: p for p in cfg["composite_detector"]["presets"]}
        assert presets_map["wacv_yolo11n"]["available"] is True
        assert presets_map["fics_yolo11n_r15"]["available"] is True

        up = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/upload",
                    data=img_p.read_bytes(),
                    headers={"X-Filename": "board_a.png"},
                    method="POST",
                ),
                timeout=10,
            ).read().decode("utf-8")
        )

        # 1) Invalid preset rejected with HTTP 400
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/jobs",
                    data=json.dumps({
                        "upload_id": up["upload_id"],
                        "model_key": "yolo11n_composite",
                        "mode": "COMPOSITE_YOLO",
                        "detector_preset": "invalid_preset",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=10,
            )
        assert exc.value.code == 400

        # 2) Run WACV then FICS R15 on exact same upload_id
        results_by_preset = {}
        for dp in ("wacv_yolo11n", "fics_yolo11n_r15"):
            job = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{base}/api/jobs",
                        data=json.dumps({
                            "upload_id": up["upload_id"],
                            "model_key": "yolo11n_composite",
                            "mode": "COMPOSITE_YOLO",
                            "detector_preset": dp,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    ),
                    timeout=10,
                ).read().decode("utf-8")
            )
            jid = job["job_id"]
            for _ in range(50):
                st = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}", timeout=10).read().decode("utf-8"))
                if st["status"] in ("completed", "failed", "incomplete"):
                    break
                time.sleep(0.05)
            assert st["status"] == "completed"
            results_by_preset[dp] = json.loads(
                urllib.request.urlopen(f"{base}/api/jobs/{jid}/result", timeout=10).read().decode("utf-8")
            )

        assert results_by_preset["wacv_yolo11n"]["kept_predictions_count"] == 1
        assert results_by_preset["fics_yolo11n_r15"]["kept_predictions_count"] == 2
        assert results_by_preset["fics_yolo11n_r15"]["composite_metadata"]["detector_preset"] == "fics_yolo11n_r15"
        assert results_by_preset["fics_yolo11n_r15"]["composite_metadata"]["yolo_weights_sha256"] == "bbb222"
        assert seen_calls == [
            ("wacv_yolo11n", "wacv_11", str(fake_wacv_pt)),
            ("fics_yolo11n_r15", "fics_6", str(fake_fics_pt)),
        ]

        # Query FICS result -> inherits fics_yolo11n_r15 preset + SHA + mechanism
        q_out = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/composite/query",
                    data=json.dumps({
                        "job_id": results_by_preset["fics_yolo11n_r15"]["job_id"],
                        "query_text": "只找二极管并计数",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=10,
            ).read().decode("utf-8")
        )
        assert q_out["predicted_count"] == 0
        assert q_out["query_execution_mechanism"] == "筛选已生成的检测结果"
        assert q_out["detector_preset"] == "fics_yolo11n_r15"
        assert q_out["yolo_weights_sha256"] == "bbb222"
    finally:
        srv.shutdown()
        srv.demo_service.shutdown()
        srv.server_close()
