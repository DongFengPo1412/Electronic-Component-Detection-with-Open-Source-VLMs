"""Round 8 tests: single-class natural-language instruction re-inference, audit contracts, and R7 progress fix."""

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
from pcb_vlm.evaluation.candidate_verifier import DEFAULT_VERIFIER_PROMPT  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402
from pcb_vlm.evaluation.language_instruction import (  # noqa: E402
    audit_single_class_predictions,
    build_single_class_detection_prompt,
    parse_pcb_language_instruction,
)


def _make_board_png(path: pathlib.Path, w: int = 1800, h: int = 1200) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[40 : h - 40, 60 : w - 60] = [170, 50, 50]
    arr[180:290, 220:360] = [210, 180, 50]
    arr[380:460, 480:560] = [50, 180, 210]
    arr[540:630, 700:810] = [70, 200, 110]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _wait_job(base_url: str, job_id: str, timeout_iters: int = 200) -> dict:
    for _ in range(timeout_iters):
        _, _, raw = _http_req(f"{base_url}/api/jobs/{job_id}")
        js = json.loads(raw.decode("utf-8"))
        if js["status"] in ("completed", "failed", "incomplete", "cancelled"):
            return js
        time.sleep(0.03)
    raise AssertionError(f"Job {job_id} did not terminate in time")


@pytest.mark.parametrize(
    "text,expected_cls,expected_count_req",
    [
        ("只找电阻并计数", "resistor", True),
        ("find all capacitors and count", "capacitor", True),
        ("标注集成电路", "ic", False),
        ("二极管", "diode", False),
        ("晶体管", "transistor", False),
        ("电感", "inductor", False),
    ],
)
def test_parse_pcb_language_instruction_supported(text: str, expected_cls: str, expected_count_req: bool) -> None:
    res = parse_pcb_language_instruction(text, coordinate_protocol="xyxy_norm1000")
    assert res["ok"] is True
    assert res["target_class"] == expected_cls
    assert res["count_requested"] is expected_count_req
    p_txt, p_sha = build_single_class_detection_prompt(expected_cls, coordinate_protocol="xyxy_norm1000")
    assert res["normalized_prompt"] == p_txt
    assert res["prompt_sha256"] == p_sha
    assert expected_cls in p_txt and f'Do NOT output any component other than {expected_cls}' in p_txt


@pytest.mark.parametrize(
    "text,expected_code",
    [
        ("", "empty_instruction"),
        ("   ", "empty_instruction"),
        ("找电阻和电容", "multi_class_not_single_target"),
        ("找所有六类器件", "multi_class_use_standard_mode"),
        ("找左上角电阻", "spatial_restriction_unsupported"),
        ("只找连接器并计数", "out_of_scope_class"),
        ("找保险丝", "out_of_scope_class"),
        ("find crystals", "out_of_scope_class"),
        ("除了电阻之外", "exclusion_instruction_unsupported"),
    ],
)
def test_parse_pcb_language_instruction_rejections(text: str, expected_code: str) -> None:
    res = parse_pcb_language_instruction(text)
    assert res["ok"] is False
    assert res["error_code"] == expected_code
    assert res["error_message"]


def test_audit_single_class_predictions_contracts() -> None:
    all_preds = [
        {"label": "resistor", "bbox_xyxy": [10, 10, 40, 20]},
        {"label": "capacitor", "bbox_xyxy": [50, 50, 80, 70]},
        {"label": "resistor", "bbox_xyxy": [100, 100, 130, 120]},
        {"label": "ic", "bbox_xyxy": [200, 200, 300, 300]},
    ]
    raw_recs = [
        {"raw_output": '[{"bbox_2d": [10, 10, 40, 20], "label": "resistor"}, {"bbox_2d": [50, 50, 80, 70], "label": "capacitor"}]'},
        {"raw_output": '[{"bbox_2d": [200, 200, 300, 300], "label": "ic"}]'},
    ]
    audited = audit_single_class_predictions(
        all_post_nms_preds=all_preds,
        raw_global_predictions=all_preds,
        raw_records=raw_recs,
        target_class="resistor",
    )
    assert audited["target_count"] == 2
    assert audited["pre_filter_post_nms_total"] == 4
    assert audited["unauthorized_post_nms_count"] == 2
    assert audited["unauthorized_post_nms_rate"] == pytest.approx(0.5)
    assert audited["unauthorized_class_breakdown"] == {"capacitor": 1, "ic": 1}
    assert audited["raw_json_unauthorized_labels"] == {"capacitor": 1, "ic": 1}

    empty_audit = audit_single_class_predictions(
        all_post_nms_preds=[],
        raw_global_predictions=[],
        raw_records=[{"raw_output": "[]"}],
        target_class="inductor",
    )
    assert empty_audit["target_count"] == 0
    assert empty_audit["unauthorized_post_nms_count"] == 0
    assert empty_audit["unauthorized_post_nms_rate"] == 0.0


def test_http_language_instruction_re_inference_same_roi_tiles_and_guards(tmp_path: pathlib.Path) -> None:
    img_bytes = _make_board_png(tmp_path / "board_r8.png", w=1800, h=1200)
    captured_plans: dict[str, dict] = {}
    runner_mode = {"kind": "normal"}

    def _mock_job_runner(job_spec: dict, cancel_ev, svc_ref: LiveDemoService) -> None:
        jdir = svc_ref.jobs_dir / job_spec["job_id"]
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        captured_plans[job_spec["job_id"]] = plan
        mk = job_spec["model_key"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gpu_gate.json").write_text(json.dumps({"passed": True, "smi_free_mib": 7000.0}), encoding="utf-8")
        q_items = plan["queues"][mk]
        lines = []
        inst = job_spec.get("instruction_spec")
        tgt = inst["target_class"] if isinstance(inst, dict) else None
        for idx, q in enumerate(q_items):
            if runner_mode["kind"] == "incomplete" and idx == len(q_items) - 1:
                continue  # Missing final expected tile!
            if runner_mode["kind"] == "valid_empty":
                objs = []
            elif idx == 0:
                if tgt == "resistor":
                    # One valid resistor matching GT [220,180,360,290] + one unauthorized capacitor!
                    objs = [
                        {"bbox_2d": [220, 180, 360, 290], "label": "resistor"},
                        {"bbox_2d": [480, 380, 560, 460], "label": "capacitor"},
                    ]
                elif tgt == "capacitor":
                    objs = [{"bbox_2d": [480, 380, 560, 460], "label": "capacitor"}]
                else:
                    objs = [
                        {"bbox_2d": [220, 180, 360, 290], "label": "resistor"},
                        {"bbox_2d": [480, 380, 560, 460], "label": "capacitor"},
                    ]
                # Map global mock coordinates into tile normalized 1000 space
                tw, th = q["width"], q["height"]
                ox, oy = q["roi_offset_x"] + q["tile_offset_x"], q["roi_offset_y"] + q["tile_offset_y"]
                mapped = []
                for o in objs:
                    x1, y1, x2, y2 = o["bbox_2d"]
                    nx1 = max(0, min(1000, round((x1 - ox) * 1000 / tw)))
                    ny1 = max(0, min(1000, round((y1 - oy) * 1000 / th)))
                    nx2 = max(0, min(1000, round((x2 - ox) * 1000 / tw)))
                    ny2 = max(0, min(1000, round((y2 - oy) * 1000 / th)))
                    mapped.append({"bbox_2d": [nx1, ny1, nx2, ny2], "label": o["label"]})
                objs = mapped
            else:
                objs = []
            raw_txt = json.dumps(objs, ensure_ascii=False) + "<|im_end|>"
            lines.append(json.dumps({
                "call_id": q["call_id"],
                "board_id": q["board_id"],
                "tag": q["tag"],
                "mode": q["mode"],
                "image_id": q["image_id"],
                "call_index": q["call_index"],
                "tile_id": q["tile_id"],
                "ok": True,
                "orig_input_width": q["width"],
                "orig_input_height": q["height"],
                "roi_offset_x": q["roi_offset_x"],
                "roi_offset_y": q["roi_offset_y"],
                "tile_offset_x": q["tile_offset_x"],
                "tile_offset_y": q["tile_offset_y"],
                "planned_qwen_min_pixels": 65536,
                "planned_qwen_max_pixels": 589824,
                "processed_width_px": q["width"],
                "processed_height_px": q["height"],
                "processed_area_px": q["width"] * q["height"],
                "image_grid_thw": [1, 48, 48],
                "merged_visual_tokens": 576,
                "max_new_tokens": 512,
                "generated_tokens": 12,
                "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000",
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "effective_generation_probe": {
                    "declared_do_sample": False,
                    "explicit_do_sample_kwarg_passed": True,
                    "effective_do_sample": False,
                    "effective_generation_mode": "GenerationMode.GREEDY_SEARCH",
                },
                "gpu_generate_ms": 25.0,
                "cuda_max_reserved_mib": 2400.0,
                "raw_output": raw_txt,
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    svc = LiveDemoService(workspace_root=tmp_path / "ws_r8", job_runner=_mock_job_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws_r8", service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        st_html, _, body_html = _http_req(f"{base}/")
        assert st_html == 200
        html_s = body_html.decode("utf-8")
        assert 'id="inpLangInstruction"' in html_s and 'id="btnStartInstructionJob"' in html_s

        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=img_bytes, headers={"X-Filename": "syn_r8.png"})
        assert st_u == 200
        up = json.loads(raw_u.decode("utf-8"))
        uid = up["upload_id"]

        # 1. Standard 6-class job
        c1, _, b1 = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c1 == 202
        jid_six = json.loads(b1.decode("utf-8"))["job_id"]
        assert _wait_job(base, jid_six)["status"] == "completed"

        # 2. Single-class instruction via /api/jobs/instruct: "只找电阻并计数"
        c2, _, b2 = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768", "user_instruction": "只找电阻并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c2 == 202
        jid_res = json.loads(b2.decode("utf-8"))["job_id"]
        assert _wait_job(base, jid_res)["status"] == "completed"

        # 3. Second instruction via /api/jobs with user_instruction: "只找电容并计数"
        c3, _, b3 = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768", "user_instruction": "只找电容并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c3 == 202
        jid_cap = json.loads(b3.decode("utf-8"))["job_id"]
        assert _wait_job(base, jid_cap)["status"] == "completed"

        # Distinct job_id, prompt_sha256, call_ids; IDENTICAL tile SHA-256 sequence!
        assert len({jid_six, jid_res, jid_cap}) == 3
        p_six, p_res, p_cap = captured_plans[jid_six], captured_plans[jid_res], captured_plans[jid_cap]
        assert len({
            p_six["inference_config"]["prompt_sha256"],
            p_res["inference_config"]["prompt_sha256"],
            p_cap["inference_config"]["prompt_sha256"],
        }) == 3
        q_six = p_six["queues"]["qwen3"]
        q_res = p_res["queues"]["qwen3"]
        q_cap = p_cap["queues"]["qwen3"]
        assert [q["sha256"] for q in q_six] == [q["sha256"] for q in q_res] == [q["sha256"] for q in q_cap]
        assert [q["tile_id"] for q in q_six] == [q["tile_id"] for q in q_res] == [q["tile_id"] for q in q_cap]
        assert set(q["call_id"] for q in q_six).isdisjoint(q["call_id"] for q in q_res)
        assert set(q["call_id"] for q in q_res).isdisjoint(q["call_id"] for q in q_cap)
        assert f"inst_{jid_res[-8:]}_resistor_" in q_res[0]["call_id"]

        # Audit unauthorized prediction & single_class_eval
        res_res = json.loads(_http_req(f"{base}/api/jobs/{jid_res}/result")[2].decode("utf-8"))
        icomp = res_res["instruction_compliance"]
        assert icomp["target_class"] == "resistor"
        assert icomp["deduplicated_target_count"] == 1
        assert icomp["pre_filter_post_nms_total"] == 2
        assert icomp["unauthorized_post_nms_count"] == 1
        assert icomp["unauthorized_post_nms_rate"] == pytest.approx(0.5)
        assert icomp["unauthorized_class_breakdown"] == {"capacitor": 1}
        assert len(res_res["predictions"]) == 1 and res_res["predictions"][0]["label"] == "resistor"
        assert len(res_res["all_pre_filter_predictions"]) == 2

        coco_gt = {
            "images": [{"id": 1, "file_name": "syn_r8.png", "width": 1800, "height": 1200}],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 2, "bbox": [220, 180, 140, 110], "area": 15400, "iscrowd": 0},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [900, 800, 80, 60], "area": 4800, "iscrowd": 0},
                {"id": 12, "image_id": 1, "category_id": 1, "bbox": [480, 380, 80, 80], "area": 6400, "iscrowd": 0},
            ],
        }
        c_gt, _, b_gt = _http_req(
            f"{base}/api/jobs/{jid_res}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": coco_gt}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_gt == 200
        eval_res = json.loads(b_gt.decode("utf-8"))["post_hoc_eval"]["single_class_eval"]
        assert eval_res["target_class"] == "resistor"
        assert eval_res["gt_count"] == 2
        assert eval_res["pred_count"] == 1
        assert eval_res["count_signed_error"] == -1
        assert eval_res["count_abs_error"] == 1
        assert eval_res["tp"] == 1 and eval_res["fp"] == 0 and eval_res["fn"] == 1
        assert eval_res["unauthorized_post_nms_count"] == 1

        md_txt = _http_req(f"{base}/api/jobs/{jid_res}/export/summary_md")[2].decode("utf-8")
        assert "独立重新推理" in md_txt and "越权输出" in md_txt

        # Out-of-scope instruction -> HTTP 400
        c_bad, _, b_bad = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768", "user_instruction": "只找连接器并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_bad == 400 and "FICS-PCB" in b_bad.decode("utf-8")

        # Valid empty (0 target objects) vs incomplete/failed job outcome distinction
        runner_mode["kind"] = "valid_empty"
        c_emp, _, b_emp = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768", "user_instruction": "只找电感并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_emp == 202
        st_emp = _wait_job(base, json.loads(b_emp.decode("utf-8"))["job_id"])
        assert st_emp["status"] == "completed" and st_emp["kept_predictions_count"] == 0
        assert st_emp["outcome"]["outcome_code"] == "completed_zero_boxes"

        runner_mode["kind"] = "incomplete"
        c_inc, _, b_inc = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768", "user_instruction": "只找电感并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_inc == 202
        st_inc = _wait_job(base, json.loads(b_inc.decode("utf-8"))["job_id"])
        assert st_inc["status"] == "incomplete" and st_inc["technical_complete"] is False
        assert st_inc["outcome"]["outcome_code"] != "completed_zero_boxes"
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()


def test_r7_dual_crop_get_job_status_progress_full_and_reused_stage1(tmp_path: pathlib.Path) -> None:
    arr = np.full((900, 1200, 3), 255, dtype=np.uint8)
    arr[60:840, 80:1120] = [170, 50, 50]
    arr[180:290, 220:360] = [210, 180, 50]
    arr[380:460, 480:560] = [50, 180, 210]
    arr[540:630, 700:810] = [70, 200, 110]
    arr[200:290, 820:910] = [190, 190, 190]
    Image.fromarray(arr).save(tmp_path / "syn_prog.png")
    img_bytes = (tmp_path / "syn_prog.png").read_bytes()

    def _mock_ver(job_spec: dict, manifest_path: pathlib.Path, cancel_ev, svc_ref: LiveDemoService) -> None:
        jdir = manifest_path.parent
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        pad = float((man.get("crop_spec") or {}).get("pad_fraction") or 0.10)
        out_dir = jdir / "verifier_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, u in enumerate(man["unique_calls"]):
            raw_out = '{"label": "other"}<|im_end|>' if abs(pad - 0.10) < 1e-6 else '{"label": "resistor"}<|im_end|>'
            s_lbl = "other" if abs(pad - 0.10) < 1e-6 else "resistor"
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
                "generated_tokens": 6,
                "has_normal_eos": True,
                "raw_output": raw_out,
                "strict_json_ok": True,
                "strict_label": s_lbl,
                "compat_regex_label": s_lbl,
                "parse_error": None,
                "gpu_generate_ms": 20.0,
                "cuda_max_reserved_mib": 3200.0,
                "ok": True,
            })
        (out_dir / "verifier_raw_outputs.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws_prog", verifier_runner=_mock_ver)
    try:
        up = svc.register_upload(raw_bytes=img_bytes, raw_filename="syn_prog.png")
        cands = [
            {"prediction_id": "p0", "orig_order": 0, "label": "capacitor", "bbox_xyxy": [220.0, 180.0, 360.0, 290.0], "score": 1.0},
            {"prediction_id": "p1", "orig_order": 1, "label": "diode", "bbox_xyxy": [480.0, 380.0, 560.0, 460.0], "score": 1.0},
            {"prediction_id": "p2", "orig_order": 2, "label": "inductor", "bbox_xyxy": [700.0, 540.0, 810.0, 630.0], "score": 1.0},
            {"prediction_id": "p3", "orig_order": 3, "label": "ic", "bbox_xyxy": [820.0, 200.0, 910.0, 290.0], "score": 1.0},
        ]
        # Full 2-pass: 4 stage1 + 4 stage2 = 8 planned == 8 completed
        sub1 = svc.submit_verifier_only_job(
            upload_id=up["upload_id"],
            source_predictions=cands,
            preset="r7_qwen3_tile768_dual_crop_consensus",
        )
        for _ in range(150):
            st1 = svc.get_job_status(sub1["job_id"])
            if st1["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.03)
        assert st1["status"] == "completed"
        assert st1["verifier_calls_planned"] == 8
        assert st1["verifier_calls_completed"] == 8
        assert st1["stage1_reused_calls"] == 0
        assert st1["stage1_new_calls_completed"] == 4
        assert st1["stage2_new_calls_completed"] == 4

        # Reused Stage-1: 4 reused, 4 new Stage-2 planned == 4 completed
        jdir1 = (tmp_path / "ws_prog" / "jobs" / sub1["job_id"])
        sub2 = svc.submit_verifier_only_job(
            upload_id=up["upload_id"],
            source_predictions=cands,
            preset="r7_qwen3_tile768_dual_crop_consensus",
            reuse_stage1={
                "manifest_path": str(jdir1 / "verifier_manifest.json"),
                "raw_outputs_path": str(jdir1 / "verifier_runs" / "verifier_raw_outputs.jsonl"),
            },
        )
        for _ in range(150):
            st2 = svc.get_job_status(sub2["job_id"])
            if st2["status"] in ("completed", "failed", "incomplete"):
                break
            time.sleep(0.03)
        assert st2["status"] == "completed"
        assert st2["verifier_calls_planned"] == 4
        assert st2["verifier_calls_completed"] == 4
        assert st2["stage1_reused_calls"] == 4
        assert st2["stage1_new_calls_completed"] == 0
        assert st2["stage2_new_calls_completed"] == 4
    finally:
        svc.shutdown()
