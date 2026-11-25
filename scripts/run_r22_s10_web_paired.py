"""GT-free Round 22 S10 live web paired A0/A1 execution, queries & pre-GT lock."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Mapping, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8826"
DEFAULT_OUTPUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration22-20261125\live_demo\s10")
DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration22-20261125\eval\r22_s10_pre_gt_lock.json")
DEFAULT_WORKSPACE_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration22-20261125\live_ws")
S10_TIF_PATH = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s10\DSLR\img\s10_front.tif")

R19_S1_QUERIES = [
    pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration19-20261125\live_demo\s1\r19_query_00.json"),
    pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration19-20261125\live_demo\s1\r19_query_01.json"),
]


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _api_json(url: str, *, method: str = "GET", body: Optional[Mapping[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload_tif(base_url: str, tif_path: pathlib.Path) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url}/api/upload",
        data=tif_path.read_bytes(),
        headers={"X-Filename": tif_path.name},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_and_poll(base_url: str, upload_id: str, detector_preset: Optional[str], timeout_s: float = 350.0) -> Dict[str, Any]:
    body: Dict[str, Any] = {"upload_id": upload_id, "model_key": "yolo11n_composite", "mode": "COMPOSITE_YOLO"}
    if detector_preset is not None:
        body["detector_preset"] = detector_preset
    sub = _api_json(f"{base_url}/api/jobs", method="POST", body=body)
    jid = str(sub["job_id"])
    t0 = time.perf_counter()
    st = sub
    while (time.perf_counter() - t0) < timeout_s:
        st = _api_json(f"{base_url}/api/jobs/{jid}")
        if st["status"] in ("completed", "failed", "cancelled", "incomplete"):
            break
        time.sleep(0.5)
    if st["status"] != "completed":
        raise RuntimeError(f"Job {jid} ({detector_preset}) failed: {st}")
    res = _api_json(f"{base_url}/api/jobs/{jid}/result")
    assert res["gt_loaded"] is False
    return {"status": st, "result": res, "wall_seconds": round(time.perf_counter() - t0, 3)}


def main() -> int:
    t_all0 = time.perf_counter()
    base_url = DEFAULT_BASE_URL
    assert _api_json(f"{base_url}/api/health").get("ok")

    cfg = _api_json(f"{base_url}/api/config")
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
    assert "fics_yolo11n_r15_r19_a1_ic_reject" in [p["preset_id"] for p in cfg["composite_detector"]["presets"]]

    if DEFAULT_OUTPUT_DIR.exists():
        shutil.rmtree(DEFAULT_OUTPUT_DIR)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    locked_files: List[Dict[str, Any]] = []

    def _lock_copy(src: pathlib.Path, dst: pathlib.Path, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        locked_files.append({"role": role, "path": str(dst), "sha256": _sha256_file(dst), "bytes": dst.stat().st_size})

    def _lock_json(dst: pathlib.Path, obj: Any, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        locked_files.append({"role": role, "path": str(dst), "sha256": _sha256_file(dst), "bytes": dst.stat().st_size})

    def _lock_download(url: str, dst: pathlib.Path, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(url, timeout=30.0).read())
        locked_files.append({"role": role, "path": str(dst), "sha256": _sha256_file(dst), "bytes": dst.stat().st_size})

    up = _upload_tif(base_url, S10_TIF_PATH)
    uid = str(up["upload_id"])
    _lock_json(DEFAULT_OUTPUT_DIR / "upload_meta.json", up, "s10_upload_meta")

    # 1) Cold-start A0 default job (omitting detector_preset)
    run_a0 = _run_and_poll(base_url, uid, detector_preset=None)
    res_a0 = run_a0["result"]
    jid_a0 = str(res_a0["job_id"])
    assert res_a0["detector_preset"] == "fics_yolo11n_r15"

    # 2) Paired A1 job on exact same upload_id
    run_a1 = _run_and_poll(base_url, uid, detector_preset="fics_yolo11n_r15_r19_a1_ic_reject")
    res_a1 = run_a1["result"]
    jid_a1 = str(res_a1["job_id"])
    assert res_a1["detector_preset"] == "fics_yolo11n_r15_r19_a1_ic_reject"

    ws_a0 = DEFAULT_WORKSPACE_DIR / "jobs" / jid_a0 / "base_runs" / "composite_yolo"
    ws_a1 = DEFAULT_WORKSPACE_DIR / "jobs" / jid_a1 / "base_runs" / "composite_yolo"

    # Verify standalone A0 default job == A1 job's underlying A0 base 100% on coords, labels, scores, order
    a0_direct = json.loads((ws_a0 / "yolo_official_025.json").read_text(encoding="utf-8"))
    a1_base = json.loads((ws_a1 / "yolo_official_025.json").read_text(encoding="utf-8"))
    assert [(p["prediction_id"], p["label"], p["score"], p["bbox_xyxy"]) for p in a0_direct] == [
        (p["prediction_id"], p["label"], p["score"], p["bbox_xyxy"]) for p in a1_base
    ], "Standalone A0 default differs from A1 base R15 detections!"

    _lock_json(DEFAULT_OUTPUT_DIR / "a0_job_result.json", res_a0, "s10_a0_job_result")
    _lock_json(DEFAULT_OUTPUT_DIR / "a1_job_result.json", res_a1, "s10_a1_job_result")
    _lock_copy(ws_a1 / "yolo_run_summary.json", DEFAULT_OUTPUT_DIR / "yolo_run_summary.json", "s10_yolo_summary")
    _lock_copy(ws_a1 / "yolo_official_025.json", DEFAULT_OUTPUT_DIR / "yolo_official_025.json", "s10_yolo_025")
    _lock_copy(ws_a1 / "yolo_low_conf_pool_005.json", DEFAULT_OUTPUT_DIR / "yolo_low_conf_pool_005.json", "s10_pool_005")
    _lock_copy(ws_a1 / "r19_a0_predictions.json", DEFAULT_OUTPUT_DIR / "r19_a0_predictions.json", "s10_a0_preds")
    _lock_copy(ws_a1 / "r19_a1_predictions.json", DEFAULT_OUTPUT_DIR / "r19_a1_predictions.json", "s10_a1_preds")
    _lock_copy(ws_a1 / "r19_postproc_audit.json", DEFAULT_OUTPUT_DIR / "r19_postproc_audit.json", "s10_a1_audit")
    _lock_download(f"{base_url}/api/jobs/{jid_a0}/export/json", DEFAULT_OUTPUT_DIR / "a0_export.json", "s10_a0_export_json")
    _lock_download(f"{base_url}/api/jobs/{jid_a0}/export/overlay_png", DEFAULT_OUTPUT_DIR / "a0_overlay.png", "s10_a0_overlay")
    _lock_download(f"{base_url}/api/jobs/{jid_a1}/export/json", DEFAULT_OUTPUT_DIR / "a1_export.json", "s10_a1_export_json")
    _lock_download(f"{base_url}/api/jobs/{jid_a1}/export/overlay_png", DEFAULT_OUTPUT_DIR / "a1_overlay.png", "s10_a1_overlay")

    queries_meta: Dict[str, Any] = {}
    for arm_tag, jid in (("a0", jid_a0), ("a1", jid_a1)):
        q_list = []
        for qi, qtxt in enumerate(("只找IC并计数", "只找电阻，但只统计左半边")):
            qr = _api_json(f"{base_url}/api/composite/query", method="POST", body={"job_id": jid, "query_text": qtxt})
            qid = str(qr["query_id"])
            _lock_json(DEFAULT_OUTPUT_DIR / f"{arm_tag}_query_{qi:02d}.json", qr, f"s10_{arm_tag}_q{qi}_json")
            _lock_download(f"{base_url}/api/composite/query_export/{jid}/{qid}", DEFAULT_OUTPUT_DIR / f"{arm_tag}_query_{qi:02d}_export.json", f"s10_{arm_tag}_q{qi}_export")
            _lock_download(f"{base_url}{qr['overlay_url']}", DEFAULT_OUTPUT_DIR / f"{arm_tag}_query_{qi:02d}_overlay.png", f"s10_{arm_tag}_q{qi}_overlay")
            q_list.append({"query_id": qid, "raw_query": qtxt, "predicted_count": qr["predicted_count"]})
        queries_meta[arm_tag] = q_list

    # Verify non-IC left-half resistor query conserved identically
    assert queries_meta["a0"][1]["predicted_count"] == queries_meta["a1"][1]["predicted_count"]

    # Verify locked R19 S1 absent-class regression evidence
    s1_reg = []
    for sp in R19_S1_QUERIES:
        qd = json.loads(sp.read_text(encoding="utf-8"))
        assert int(qd["predicted_count"]) == 0
        s1_reg.append({"path": str(sp), "sha256": _sha256_file(sp), "raw_query": qd["query_spec"]["raw_query"], "predicted_count": qd["predicted_count"]})

    lock_doc = {
        "schema_version": "r22_s10_pre_gt_lock_v1",
        "base_url": base_url,
        "gt_loaded_during_inference": False,
        "total_wall_seconds": round(time.perf_counter() - t_all0, 3),
        "s10": {
            "upload_id": uid,
            "tif_path": str(S10_TIF_PATH),
            "tif_sha256": _sha256_file(S10_TIF_PATH),
            "roi_bbox_half_open": up["roi_bbox_half_open"],
            "a0_job_id": jid_a0,
            "a0_wall_seconds": run_a0["wall_seconds"],
            "a1_job_id": jid_a1,
            "a1_wall_seconds": run_a1["wall_seconds"],
            "queries": queries_meta,
        },
        "s1_absent_regression_verified": s1_reg,
        "locked_files": locked_files,
    }
    DEFAULT_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOCK_FILE.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE S10 pre-GT lock: {len(locked_files)} files, jid_a0={jid_a0}, jid_a1={jid_a1}, queries={queries_meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
