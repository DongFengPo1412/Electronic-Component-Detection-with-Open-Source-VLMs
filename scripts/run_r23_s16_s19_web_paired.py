"""GT-free Round 23 S16+S19 live web paired A0/A1 execution, queries & pre-GT lock."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Mapping, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8827"
DEFAULT_OUTPUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration23-20261125\live_demo")
DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration23-20261125\eval\r23_s16_s19_pre_gt_lock.json")
DEFAULT_WORKSPACE_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration23-20261125\live_ws")

BOARDS = [
    ("s16", pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s16\DSLR\img\s16_front.tif")),
    ("s19", pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s19\DSLR\img\s19_front.tif")),
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


def _run_and_poll(base_url: str, upload_id: str, detector_preset: Optional[str], timeout_s: float = 200.0) -> Dict[str, Any]:
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
    gpu_runs_log: List[Dict[str, Any]] = []
    gpu_wall_total = 0.0
    boards_meta: Dict[str, Any] = {}

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

    for bkey, tif_p in BOARDS:
        bdir = DEFAULT_OUTPUT_DIR / bkey
        up = _upload_tif(base_url, tif_p)
        uid = str(up["upload_id"])
        _lock_json(bdir / "upload_meta.json", up, f"{bkey}_upload_meta")

        # 1) Standalone A0 default job
        run_a0 = _run_and_poll(base_url, uid, detector_preset=None)
        res_a0 = run_a0["result"]
        jid_a0 = str(res_a0["job_id"])
        assert res_a0["detector_preset"] == "fics_yolo11n_r15"
        gpu_wall_total += run_a0["wall_seconds"]
        gpu_runs_log.append({"board": bkey, "arm": "A0", "job_id": jid_a0, "wall_seconds": run_a0["wall_seconds"]})

        # 2) Paired A1 job on exact same upload_id
        run_a1 = _run_and_poll(base_url, uid, detector_preset="fics_yolo11n_r15_r19_a1_ic_reject")
        res_a1 = run_a1["result"]
        jid_a1 = str(res_a1["job_id"])
        assert res_a1["detector_preset"] == "fics_yolo11n_r15_r19_a1_ic_reject"
        gpu_wall_total += run_a1["wall_seconds"]
        gpu_runs_log.append({"board": bkey, "arm": "A1", "job_id": jid_a1, "wall_seconds": run_a1["wall_seconds"]})

        assert gpu_wall_total <= 400.0, f"Exceeded 400s GPU cap: {gpu_wall_total}"

        ws_a0 = DEFAULT_WORKSPACE_DIR / "jobs" / jid_a0 / "base_runs" / "composite_yolo"
        ws_a1 = DEFAULT_WORKSPACE_DIR / "jobs" / jid_a1 / "base_runs" / "composite_yolo"

        sum_a0 = json.loads((ws_a0 / "yolo_run_summary.json").read_text(encoding="utf-8"))
        sum_a1 = json.loads((ws_a1 / "yolo_run_summary.json").read_text(encoding="utf-8"))
        assert sum_a0["geometry"]["tile_count"] == sum_a1["geometry"]["tile_count"]
        assert sum_a0["geometry"]["roi_bbox_half_open"] == sum_a1["geometry"]["roi_bbox_half_open"]

        off_a0 = json.loads((ws_a0 / "yolo_official_025.json").read_text(encoding="utf-8"))
        off_a1 = json.loads((ws_a1 / "yolo_official_025.json").read_text(encoding="utf-8"))
        assert [(p["prediction_id"], p["label"], p["score"], p["bbox_xyxy"]) for p in off_a0] == [
            (p["prediction_id"], p["label"], p["score"], p["bbox_xyxy"]) for p in off_a1
        ], f"{bkey}: A0 and A1 base official detections differ!"

        pool_a0 = json.loads((ws_a0 / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        pool_a1 = json.loads((ws_a1 / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        assert len(pool_a0) == len(pool_a1)

        _lock_json(bdir / "a0_job_result.json", res_a0, f"{bkey}_a0_job_result")
        _lock_json(bdir / "a1_job_result.json", res_a1, f"{bkey}_a1_job_result")
        _lock_copy(ws_a1 / "yolo_run_summary.json", bdir / "yolo_run_summary.json", f"{bkey}_yolo_summary")
        _lock_copy(ws_a1 / "yolo_official_025.json", bdir / "yolo_official_025.json", f"{bkey}_yolo_025")
        _lock_copy(ws_a1 / "yolo_low_conf_pool_005.json", bdir / "yolo_low_conf_pool_005.json", f"{bkey}_pool_005")
        _lock_copy(ws_a1 / "r19_a0_predictions.json", bdir / "r19_a0_predictions.json", f"{bkey}_a0_preds")
        _lock_copy(ws_a1 / "r19_a1_predictions.json", bdir / "r19_a1_predictions.json", f"{bkey}_a1_preds")
        _lock_copy(ws_a1 / "r19_postproc_audit.json", bdir / "r19_postproc_audit.json", f"{bkey}_a1_audit")
        _lock_download(f"{base_url}/api/jobs/{jid_a0}/export/json", bdir / "a0_export.json", f"{bkey}_a0_export_json")
        _lock_download(f"{base_url}/api/jobs/{jid_a0}/export/overlay_png", bdir / "a0_overlay.png", f"{bkey}_a0_overlay")
        _lock_download(f"{base_url}/api/jobs/{jid_a1}/export/json", bdir / "a1_export.json", f"{bkey}_a1_export_json")
        _lock_download(f"{base_url}/api/jobs/{jid_a1}/export/overlay_png", bdir / "a1_overlay.png", f"{bkey}_a1_overlay")

        q_meta: Dict[str, Any] = {}
        for arm_tag, jid in (("a0", jid_a0), ("a1", jid_a1)):
            ql = []
            for qi, qtxt in enumerate(("只找IC并计数", "只找电阻并计数")):
                qr = _api_json(f"{base_url}/api/composite/query", method="POST", body={"job_id": jid, "query_text": qtxt})
                assert qr["gpu_called_for_query"] is False
                qid = str(qr["query_id"])
                _lock_json(bdir / f"{arm_tag}_query_{qi:02d}.json", qr, f"{bkey}_{arm_tag}_q{qi}_json")
                _lock_download(f"{base_url}/api/composite/query_export/{jid}/{qid}", bdir / f"{arm_tag}_query_{qi:02d}_export.json", f"{bkey}_{arm_tag}_q{qi}_export")
                _lock_download(f"{base_url}{qr['overlay_url']}", bdir / f"{arm_tag}_query_{qi:02d}_overlay.png", f"{bkey}_{arm_tag}_q{qi}_overlay")
                ql.append({"query_id": qid, "raw_query": qtxt, "predicted_count": qr["predicted_count"]})
            q_meta[arm_tag] = ql

        assert q_meta["a0"][1]["predicted_count"] == q_meta["a1"][1]["predicted_count"]
        boards_meta[bkey] = {
            "upload_id": uid,
            "tif_path": str(tif_p),
            "tif_sha256": _sha256_file(tif_p),
            "roi_bbox_half_open": up["roi_bbox_half_open"],
            "tile_count": sum_a1["geometry"]["tile_count"],
            "a0_job_id": jid_a0,
            "a1_job_id": jid_a1,
            "a0_wall_seconds": run_a0["wall_seconds"],
            "a1_wall_seconds": run_a1["wall_seconds"],
            "queries": q_meta,
            "artifacts_dir": str(bdir),
        }
        print(f"[LOCKED {bkey}] a0={jid_a0} ({run_a0['wall_seconds']}s), a1={jid_a1} ({run_a1['wall_seconds']}s), queries={q_meta}", flush=True)

    lock_doc = {
        "schema_version": "r23_s16_s19_pre_gt_lock_v1",
        "base_url": base_url,
        "gt_loaded_during_inference": False,
        "gpu_wall_seconds_total": round(gpu_wall_total, 3),
        "gpu_runs_log": gpu_runs_log,
        "total_wall_seconds": round(time.perf_counter() - t_all0, 3),
        "boards": boards_meta,
        "locked_files": locked_files,
    }
    DEFAULT_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOCK_FILE.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE R23 pre-GT lock: {len(locked_files)} files, gpu_wall={gpu_wall_total:.2f}s -> {DEFAULT_LOCK_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
