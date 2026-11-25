"""CPU unit and HTTP integration tests for Round 16 FICS default promotion, export provenance, and IC calibration."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (_SRC_ROOT, _SCRIPTS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_COMPOSITE_DETECTOR_PRESET,
    EXPECTED_FICS_R15_BEST_SHA256,
    YOLO_FICS6_ID_TO_RAW_NAME,
    YOLO_WACV_ID_TO_RAW_NAME,
    filter_fics6_pool_by_class_thresholds,
    normalize_yolo_raw_class,
    split_aligned_and_extra_predictions,
)
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402
import fics_final_demo  # noqa: E402


def _make_board_png(path: pathlib.Path, size: tuple[int, int] = (400, 300)) -> None:
    im = Image.new("RGB", size, (18, 92, 54))
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 40, 100, 80], fill=(220, 180, 70))
    draw.rectangle([250, 100, 360, 220], fill=(70, 150, 220))
    im.save(path)


def test_ic_only_threshold_filter_preserves_other_five_classes() -> None:
    pool = [
        {"prediction_id": "c1", "label": "capacitor", "aligned_fics6_label": "capacitor", "is_fics6_aligned": True, "score": 0.26, "bbox_xyxy": [10, 10, 20, 20]},
        {"prediction_id": "r1", "label": "resistor", "aligned_fics6_label": "resistor", "is_fics6_aligned": True, "score": 0.25, "bbox_xyxy": [30, 10, 40, 20]},
        {"prediction_id": "d1", "label": "diode", "aligned_fics6_label": "diode", "is_fics6_aligned": True, "score": 0.25, "bbox_xyxy": [50, 10, 60, 20]},
        {"prediction_id": "ic_low", "label": "ic", "aligned_fics6_label": "ic", "is_fics6_aligned": True, "score": 0.34, "bbox_xyxy": [100, 100, 180, 180]},
        {"prediction_id": "ic_high", "label": "ic", "aligned_fics6_label": "ic", "is_fics6_aligned": True, "score": 0.56, "bbox_xyxy": [200, 100, 280, 180]},
    ]
    kept = filter_fics6_pool_by_class_thresholds(pool, default_conf=0.25, per_class_conf={"ic": 0.45})
    ids = [x["prediction_id"] for x in kept]
    assert ids == ["c1", "r1", "d1", "ic_high"]

    sp = split_aligned_and_extra_predictions(pool, conf_official=0.25, per_class_conf={"ic": 0.45})
    assert [x["prediction_id"] for x in sp["official_predictions_fics6"]] == ["c1", "r1", "d1", "ic_high"]
    assert len(sp["low_conf_pool_fics6"]) == 5


def test_four_point_fics_r15_default_and_exports(tmp_path: pathlib.Path) -> None:
    ws = tmp_path / "ws"
    img_p = tmp_path / "board_r16.png"
    _make_board_png(img_p, (400, 300))

    fake_fics_pt = tmp_path / "fics_r15_best.pt"
    fake_fics_pt.write_bytes(b"mock-fics-pt")
    fake_wacv_pt = tmp_path / "wacv_best.pt"
    fake_wacv_pt.write_bytes(b"mock-wacv-pt")

    seen_plans = []

    def fake_yolo(job_spec, plan, comp_out, cancel_ev, service):
        yolo_m = plan["models"]["yolo11n"]
        dp = yolo_m["detector_preset"]
        seen_plans.append(dict(yolo_m))
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
                "gpu_generate_ms": 3.0,
            }
            for q in q_items
        ]
        if dp == "wacv_yolo11n":
            info = normalize_yolo_raw_class(1, YOLO_WACV_ID_TO_RAW_NAME, expected_schema="wacv_11")
            preds = [{
                "prediction_id": "w1", "board_id": job_spec["board_id"], "image_id": 1,
                "label": "resistor", "score": 0.61, "bbox_xyxy": [40.0, 40.0, 100.0, 80.0],
                "detector_source": "yolo11n", "detector_preset": dp, "vlm_intervened": False, **info,
            }]
        else:
            info_res = normalize_yolo_raw_class(1, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
            info_ic = normalize_yolo_raw_class(5, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
            preds = [
                {
                    "prediction_id": "f1", "board_id": job_spec["board_id"], "image_id": 1,
                    "label": "resistor", "score": 0.90, "bbox_xyxy": [40.0, 40.0, 100.0, 80.0],
                    "detector_source": dp, "detector_preset": dp, "vlm_intervened": False, **info_res,
                },
                {
                    "prediction_id": "f2", "board_id": job_spec["board_id"], "image_id": 1,
                    "label": "ic", "score": 0.35, "bbox_xyxy": [250.0, 100.0, 360.0, 220.0],
                    "detector_source": dp, "detector_preset": dp, "vlm_intervened": False, **info_ic,
                },
            ]
            if yolo_m.get("per_class_conf"):
                preds = filter_fics6_pool_by_class_thresholds(preds, default_conf=0.25, per_class_conf=yolo_m["per_class_conf"])

        per_cls_thr = {c: float((yolo_m.get("per_class_conf") or {}).get(c, 0.25)) for c in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic")}
        return {
            "summary": {
                "detector_preset": dp,
                "expected_class_schema": yolo_m["expected_class_schema"],
                "weights_sha256": yolo_m["weights_sha256"],
                "qwen3_participated": False,
                "operating_point": {"class_conf_thresholds": per_cls_thr},
                "counts": {"pre_nms_all": len(preds)},
                "timing_and_vram": {"cuda_max_memory_reserved_mib": 88.0},
            },
            "low_conf_pool_fics6": preds,
            "official_predictions_fics6": preds,
            "extra_info": {"counts_005": {}, "counts_025": {}},
            "raw_records": raw_records,
        }

    rt_cfg_path = tmp_path / "runtime_config.yaml"
    rt_cfg = {
        "default_composite_detector_preset": "fics_yolo11n_r15",
        "models": {
            "yolo11n": {"local_dir": str(fake_wacv_pt), "weights_sha256": "wacv111"},
            "yolo11n_fics_r15": {"local_dir": str(fake_fics_pt), "weights_sha256": EXPECTED_FICS_R15_BEST_SHA256},
            "yolo11n_fics_r15_ic_cal": {
                "local_dir": str(fake_fics_pt),
                "weights_sha256": EXPECTED_FICS_R15_BEST_SHA256,
                "ic_conf_threshold": 0.45,
            },
        },
    }
    rt_cfg_path.write_text(yaml.safe_dump(rt_cfg), encoding="utf-8")
    svc = LiveDemoService(workspace_root=ws, runtime_config_path=rt_cfg_path, composite_yolo_runner=fake_yolo)
    srv = fics_final_demo.create_demo_server(workspace_root=ws, host="127.0.0.1", port=0, service=svc)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # 1) HTML cold-start default option
        html = urllib.request.urlopen(f"{base}/", timeout=10).read().decode("utf-8")
        assert '<option value="fics_yolo11n_r15" selected>' in html

        # 2) /api/config default
        cfg = json.loads(urllib.request.urlopen(f"{base}/api/config", timeout=10).read().decode("utf-8"))
        assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
        assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
        assert cfg["composite_detector"]["presets"][0]["preset_id"] == "fics_yolo11n_r15"
        assert cfg["composite_detector"]["presets"][0]["weights_sha256"] == EXPECTED_FICS_R15_BEST_SHA256
        assert cfg["single_model_defense"]["model_key"] == "qwen3"
        assert cfg["three_model_compare"]["mode"] == "W"

        up = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/upload",
                    data=img_p.read_bytes(),
                    headers={"X-Filename": "board_r16.png"},
                    method="POST",
                ),
                timeout=10,
            ).read().decode("utf-8")
        )

        # 3) Submit COMPOSITE_YOLO OMITTING detector_preset -> must select fics_yolo11n_r15 + SHA
        job = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/jobs",
                    data=json.dumps({
                        "upload_id": up["upload_id"],
                        "model_key": "yolo11n_composite",
                        "mode": "COMPOSITE_YOLO",
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
        assert seen_plans[0]["detector_preset"] == "fics_yolo11n_r15"
        assert seen_plans[0]["weights_sha256"] == EXPECTED_FICS_R15_BEST_SHA256
        assert seen_plans[0]["local_dir"] == str(fake_fics_pt)

        # 4) Verify exports include weight_identity, filter_source, known_limitations, and qwen3_participated=False
        exp_json = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}/export/json", timeout=10).read().decode("utf-8"))
        assert exp_json["composite_metadata"]["detector_preset"] == "fics_yolo11n_r15"
        assert exp_json["composite_metadata"]["qwen3_participated"] is False
        assert exp_json["export_provenance"]["weight_identity"]["weights_sha256"] == EXPECTED_FICS_R15_BEST_SHA256
        assert exp_json["export_provenance"]["filter_source"] == "composite_yolo_direct_025"
        assert exp_json["export_provenance"]["qwen3_participated"] is False
        assert len(exp_json["export_provenance"]["known_limitations"]) >= 1

        exp_md = urllib.request.urlopen(f"{base}/api/jobs/{jid}/export/summary_md", timeout=10).read().decode("utf-8")
        assert "fics_yolo11n_r15" in exp_md
        assert "Qwen3 参与类别输出：`False`" in exp_md

        # Query & query export
        q_out = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/composite/query",
                    data=json.dumps({"job_id": jid, "query_text": "只找电阻，但只统计左半边"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=10,
            ).read().decode("utf-8")
        )
        q_exp = json.loads(
            urllib.request.urlopen(
                f"{base}/api/composite/query_export/{jid}/{q_out['query_id']}",
                timeout=10,
            ).read().decode("utf-8")
        )
        assert q_exp["weight_identity"]["detector_preset"] == "fics_yolo11n_r15"
        assert q_exp["filter_source"] == "筛选已生成的检测结果"
        assert q_exp["qwen3_participated"] is False
        assert q_exp["predictions"][0]["bbox_xyxy"] == [40.0, 40.0, 100.0, 80.0]

        # 5) Switch to experimental IC-cal preset -> filters out ic score=0.35 while keeping resistor=0.90
        job_ic = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/jobs",
                    data=json.dumps({
                        "upload_id": up["upload_id"],
                        "model_key": "yolo11n_composite",
                        "mode": "COMPOSITE_YOLO",
                        "detector_preset": "fics_yolo11n_r15_ic_cal",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=10,
            ).read().decode("utf-8")
        )
        jid_ic = job_ic["job_id"]
        for _ in range(50):
            st2 = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid_ic}", timeout=10).read().decode("utf-8"))
            if st2["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.05)
        assert st2["status"] == "completed"
        res_ic = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid_ic}/result", timeout=10).read().decode("utf-8"))
        assert res_ic["kept_predictions_count"] == 1
        assert res_ic["export_provenance"]["filter_source"] == "composite_yolo_ic_calibrated"
    finally:
        srv.shutdown()
        srv.demo_service.shutdown()
        srv.server_close()
