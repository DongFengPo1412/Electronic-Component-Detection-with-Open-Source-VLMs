"""Round 18 unit & HTTP integration tests for frozen-locator + supervised IC/D/Q crop verifier."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw
import torch
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fics_final_demo  # noqa: E402
import prepare_r18_crop_dataset as r18_prep  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_COMPOSITE_DETECTOR_PRESET,
    EXPECTED_FICS_R15_BEST_SHA256,
    SUPPORTED_COMPOSITE_DETECTOR_PRESETS,
    YOLO_FICS6_ID_TO_RAW_NAME,
    normalize_yolo_raw_class,
)
from pcb_vlm.evaluation.crop_verifier_r18 import (  # noqa: E402
    R18_CLASSES,
    apply_r18_verifier_to_predictions,
    decide_r18_action,
    extract_square_context_crop,
)
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402


class _StubModel(torch.nn.Module):
    """Deterministic stub returning preset logits per crop order."""

    def __init__(self, logits_rows: list[list[float]]) -> None:
        super().__init__()
        self._rows = logits_rows
        self._cursor = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x.shape[0])
        out = torch.tensor(self._rows[self._cursor : self._cursor + n], dtype=torch.float32, device=x.device)
        self._cursor += n
        return out


def test_r18_abstention_and_non_target_invariance(tmp_path: pathlib.Path) -> None:
    img_p = tmp_path / "board.png"
    im = Image.new("RGB", (300, 200), (20, 90, 50))
    ImageDraw.Draw(im).rectangle([20, 20, 80, 60], fill=(200, 180, 90))
    im.save(img_p)

    # Pure RGB crop size & zero crash on border box
    crop = extract_square_context_crop(im, [0.0, 0.0, 20.0, 15.0])
    assert crop.size == (224, 224)

    # Abstention rule: below t_relabel=0.85 -> abstain_keep_orig
    act_abs = decide_r18_action(
        orig_label="ic",
        probs_by_class={"ic": 0.20, "diode": 0.05, "transistor": 0.70, "other": 0.05},
        t_relabel=0.85, m_relabel=0.15, t_reject_other=0.75, m_reject_other=0.25,
    )
    assert act_abs["decision"] == "abstain_keep_orig" and act_abs["new_label"] == "ic"

    preds = [
        {"prediction_id": "p_cap", "label": "capacitor", "aligned_fics6_label": "capacitor", "score": 0.91, "bbox_xyxy": [10.0, 10.0, 30.0, 25.0]},
        {"prediction_id": "p_res", "label": "resistor", "aligned_fics6_label": "resistor", "score": 0.88, "bbox_xyxy": [40.0, 10.0, 60.0, 25.0]},
        {"prediction_id": "p_ic1", "label": "ic", "aligned_fics6_label": "ic", "score": 0.77, "bbox_xyxy": [80.0, 40.0, 130.0, 80.0]},
        {"prediction_id": "p_ic2", "label": "ic", "aligned_fics6_label": "ic", "score": 0.65, "bbox_xyxy": [150.0, 40.0, 210.0, 90.0]},
        {"prediction_id": "p_ic3", "label": "ic", "aligned_fics6_label": "ic", "score": 0.42, "bbox_xyxy": [220.0, 40.0, 260.0, 80.0]},
    ]
    # p_ic1 -> high-conf transistor; p_ic2 -> high-conf other (reject); p_ic3 -> uncertain (abstain)
    stub = _StubModel([
        [-2.0, -2.0, 4.0, -2.0],   # softmax ~ 0.99 transistor
        [-2.0, -2.0, -2.0, 4.0],   # softmax ~ 0.99 other
        [0.3, 0.2, 0.4, 0.1],      # uncertain -> abstain
    ])
    out = apply_r18_verifier_to_predictions(
        image_path=img_p,
        predictions=preds,
        model=stub,
        device=torch.device("cpu"),
        weights_sha256="abc123",
        t_relabel=0.85,
        m_relabel=0.15,
        t_reject_other=0.75,
        m_reject_other=0.25,
    )
    vp = out["verified_predictions"]
    assert len(vp) == 4  # p_ic2 rejected
    # Non-targets 100% bit-for-bit identical in bbox & score & label
    assert vp[0]["label"] == "capacitor" and vp[0]["bbox_xyxy"] == [10.0, 10.0, 30.0, 25.0] and vp[0]["score"] == 0.91
    assert vp[1]["label"] == "resistor" and vp[1]["bbox_xyxy"] == [40.0, 10.0, 60.0, 25.0] and vp[1]["score"] == 0.88
    # Relabeled transistor keeps exact R15 bbox & score
    assert vp[2]["label"] == "transistor" and vp[2]["orig_r15_label"] == "ic" and vp[2]["bbox_xyxy"] == [80.0, 40.0, 130.0, 80.0] and vp[2]["score"] == 0.77
    assert vp[3]["label"] == "ic" and vp[3]["verifier_decision"] == "abstain_keep_orig"
    assert out["summary"]["verifier_calls"] == 3


def test_r18_banned_boards_and_http_service_preset(tmp_path: pathlib.Path) -> None:
    assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
    assert "fics_yolo11n_r15_crop_verify_r18" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS
    required_banned = {"s20", "s4", "s1", "s12", "s24", "s31", "s10", "s16", "s19", "s9", "s17", "s11", "s5", "s6", "s7"}
    assert required_banned.issubset(r18_prep.BANNED_BOARDS)

    ws = tmp_path / "ws"
    img_p = tmp_path / "b18.png"
    im = Image.new("RGB", (400, 300), (18, 92, 54))
    ImageDraw.Draw(im).rectangle([40, 40, 100, 80], fill=(220, 180, 70))
    im.save(img_p)

    r15_pt = tmp_path / "r15.pt"
    r15_pt.write_bytes(b"r15")
    cv_pt = tmp_path / "cv18.pt"
    cv_pt.write_bytes(b"cv18")

    def fake_yolo(job_spec, plan, comp_out, cancel_ev, service):
        q = plan["queues"]["yolo11n"][0]
        raw_records = [{"call_id": q["call_id"], "call_index": 0, "board_id": job_spec["board_id"], "tile_id": 0, "width": q["width"], "height": q["height"], "parsed_count": 2, "raw_output": "[]", "gpu_generate_ms": 2.0}]
        i_res = normalize_yolo_raw_class(1, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
        i_ic = normalize_yolo_raw_class(5, YOLO_FICS6_ID_TO_RAW_NAME, expected_schema="fics_6")
        preds = [
            {"prediction_id": "r1", "board_id": job_spec["board_id"], "image_id": 1, "label": "resistor", "score": 0.81, "bbox_xyxy": [40.0, 40.0, 80.0, 70.0], "detector_source": "yolo11n", "detector_preset": job_spec["detector_preset"], "vlm_intervened": False, **i_res},
            {"prediction_id": "ic1", "board_id": job_spec["board_id"], "image_id": 1, "label": "ic", "score": 0.72, "bbox_xyxy": [140.0, 80.0, 200.0, 120.0], "detector_source": "yolo11n", "detector_preset": job_spec["detector_preset"], "vlm_intervened": False, **i_ic},
        ]
        return {
            "raw_records": raw_records,
            "official_predictions_fics6": preds,
            "low_conf_pool_fics6": list(preds),
            "extra_predictions": {"counts_025": {}, "counts_005": {}},
            "summary": {"detector_preset": job_spec["detector_preset"], "weights_sha256": EXPECTED_FICS_R15_BEST_SHA256, "qwen3_participated": False, "counts": {"pre_nms_all": 2, "low_conf_pool_005_fics6": 2, "official_025_fics6": 2, "non_fics_class_counts_025": {}, "non_fics_class_counts_005": {}}, "geometry": {"roi_bbox_half_open": [0, 0, 400, 300], "tile_count": 1}, "timing_and_vram": {"total_wall_seconds": 0.1}},
        }

    rt_cfg_p = tmp_path / "rt.yaml"
    rt_cfg_p.write_text(yaml.safe_dump({
        "default_composite_detector_preset": "fics_yolo11n_r15",
        "models": {
            "yolo11n_fics_r15": {"local_dir": str(r15_pt), "weights_sha256": EXPECTED_FICS_R15_BEST_SHA256},
            "yolo11n_fics_r15_crop_verify_r18": {"local_dir": str(r15_pt), "weights_sha256": EXPECTED_FICS_R15_BEST_SHA256, "crop_verifier_weights": str(cv_pt), "crop_verifier_sha256": "deadbeef"},
        },
    }), encoding="utf-8")
    svc = LiveDemoService(workspace_root=ws, runtime_config_path=rt_cfg_p, composite_yolo_runner=fake_yolo)
    svc._custom_r18_crop_verifier = lambda *, image_path, predictions, det_spec: {  # type: ignore[attr-defined]
        "verified_predictions": [
            dict(predictions[0], orig_r15_label="resistor", verifier_decision="untouched_non_target", verifier_source="frozen_r15_passthrough"),
            dict(predictions[1], label="transistor", aligned_fics6_label="transistor", orig_r15_label="ic", verifier_decision="relabel", verifier_source="r18_resnet18_crop_verifier"),
        ],
        "decisions_log": [{"orig_r15_label": "ic", "verified_label": "transistor", "decision": "relabel"}],
        "summary": {"verifier_calls": 1, "verifier_wall_seconds": 0.01, "decision_counts": {"relabel": 1}},
    }

    srv = fics_final_demo.create_demo_server(workspace_root=ws, host="127.0.0.1", port=0, service=svc)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        up = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/api/upload", data=img_p.read_bytes(), headers={"X-Filename": "b18.png"}, method="POST"), timeout=10).read().decode("utf-8"))
        sub = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/api/jobs", data=json.dumps({"upload_id": up["upload_id"], "model_key": "yolo11n_composite", "mode": "COMPOSITE_YOLO", "detector_preset": "fics_yolo11n_r15_crop_verify_r18"}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"), timeout=10).read().decode("utf-8"))
        jid = sub["job_id"]
        st = {}
        for _ in range(60):
            st = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}", timeout=10).read().decode("utf-8"))
            if st["status"] in ("completed", "failed", "cancelled", "incomplete"):
                break
            time.sleep(0.1)
        assert st.get("status") == "completed", f"Job failed: {st}"
        res = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{jid}/result", timeout=10).read().decode("utf-8"))
        assert res["detector_preset"] == "fics_yolo11n_r15_crop_verify_r18"
        assert res["composite_metadata"]["r18_crop_verification"]["verifier_calls"] == 1
        assert [p["label"] for p in res["predictions"]] == ["resistor", "transistor"]
    finally:
        srv.shutdown()
        srv.server_close()
