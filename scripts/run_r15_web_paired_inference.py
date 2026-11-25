"""Pure GT-free Round 15 real web upload, paired inference, language queries & pre-GT lock script.

Strict separation:
- ZERO Ground Truth / COCO file imports or reads anywhere in this file.
- Connects to the running web server (`http://127.0.0.1:8819`), uploads original TIF boards (`S20`, `S4`, `S1`),
  runs BOTH `wacv_yolo11n` and `fics_yolo11n_r15` on each uploaded image (`upload_id`) for 100% paired ROI/tiling,
  runs composite natural-language / spatial / absent-class queries via `/api/composite/query`,
  persists raw outputs to `--output-dir`, and writes `--lock-file` with SHA-256s BEFORE any post-GT scoring runs.
- Strictly refuses to touch S31, S10, s16, s19, or S9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8819"
DEFAULT_OUTPUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\live_demo")
DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\eval\r15_pre_gt_lock.json")

BOARDS_SPEC: List[Dict[str, Any]] = [
    {
        "board_key": "s20",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif",
        "queries": [
            "只找电阻并计数",
            "只找电阻，但只统计左半边",
            "找电阻和电容",
        ],
    },
    {
        "board_key": "s4",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif",
        "queries": [
            "只找电容并计数",
        ],
    },
    {
        "board_key": "s1",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\img\s1_front.tif",
        "queries": [
            "只找电阻，但只统计左半边",
            "只找二极管并计数",
            "只找晶体管并计数",
        ],
    },
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


def _run_and_poll_composite(
    *,
    base_url: str,
    upload_id: str,
    detector_preset: str,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    sub = _api_json(
        f"{base_url}/api/jobs",
        method="POST",
        body={
            "upload_id": upload_id,
            "model_key": "yolo11n_composite",
            "mode": "COMPOSITE_YOLO",
            "detector_preset": detector_preset,
        },
    )
    jid = str(sub["job_id"])
    t0 = time.perf_counter()
    st = sub
    while (time.perf_counter() - t0) < timeout_s:
        st = _api_json(f"{base_url}/api/jobs/{jid}")
        if st["status"] in ("completed", "failed", "cancelled", "incomplete"):
            break
        time.sleep(0.5)
    if st["status"] != "completed":
        raise RuntimeError(f"Composite job {jid} ({detector_preset}) failed: {st}")
    res = _api_json(f"{base_url}/api/jobs/{jid}/result")
    assert res["gt_loaded"] is False, f"GT must NOT be loaded during live inference! {jid}"
    return {"status": st, "result": res, "wall_seconds": round(time.perf_counter() - t0, 3)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run GT-free paired composite web inference and lock predictions.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lock-file", type=pathlib.Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--workspace-dir", type=pathlib.Path, default=pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\live_ws"))
    args = parser.parse_args(argv)

    t_all0 = time.perf_counter()
    base_url = args.base_url.rstrip("/")
    health = _api_json(f"{base_url}/api/health")
    assert health.get("ok"), f"Health check failed: {health}"

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locked_files: List[Dict[str, Any]] = []
    board_records: Dict[str, Any] = {}

    def _copy_and_lock(src: pathlib.Path, dst: pathlib.Path, role: str) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha = _sha256_file(dst)
        locked_files.append({"role": role, "path": str(dst), "sha256": sha, "bytes": dst.stat().st_size})
        return sha

    def _write_and_lock_json(dst: pathlib.Path, obj: Any, role: str) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sha = _sha256_file(dst)
        locked_files.append({"role": role, "path": str(dst), "sha256": sha, "bytes": dst.stat().st_size})
        return sha

    for bspec in BOARDS_SPEC:
        bkey = str(bspec["board_key"])
        tif_p = pathlib.Path(str(bspec["tif_path"]))
        assert bkey not in ("s31", "s10", "s16", "s19", "s9"), f"Banned board in inference spec: {bkey}"
        up = _upload_tif(base_url, tif_p)
        uid = str(up["upload_id"])
        b_dir = args.output_dir / bkey
        _write_and_lock_json(b_dir / "upload_meta.json", up, f"{bkey}_upload_meta")

        arms_out: Dict[str, Any] = {}
        for dp in ("wacv_yolo11n", "fics_yolo11n_r15"):
            run = _run_and_poll_composite(base_url=base_url, upload_id=uid, detector_preset=dp)
            res = run["result"]
            jid = str(res["job_id"])
            a_dir = b_dir / dp
            j_ws_dir = args.workspace_dir / "jobs" / jid
            comp_ws_dir = j_ws_dir / "base_runs" / "composite_yolo"

            _write_and_lock_json(a_dir / "job_result.json", res, f"{bkey}_{dp}_job_result")
            _copy_and_lock(comp_ws_dir / "yolo_raw_outputs_all.jsonl", a_dir / "yolo_raw_outputs_all.jsonl", f"{bkey}_{dp}_raw_jsonl")
            _copy_and_lock(comp_ws_dir / "yolo_official_025.json", a_dir / "yolo_official_025.json", f"{bkey}_{dp}_official_025")
            _copy_and_lock(comp_ws_dir / "yolo_low_conf_pool_005.json", a_dir / "yolo_low_conf_pool_005.json", f"{bkey}_{dp}_pool_005")
            _copy_and_lock(comp_ws_dir / "yolo_run_summary.json", a_dir / "yolo_run_summary.json", f"{bkey}_{dp}_run_summary")
            _copy_and_lock(comp_ws_dir / "yolo_raw_non_fics_extra.json", a_dir / "yolo_raw_non_fics_extra.json", f"{bkey}_{dp}_extra")

            arms_out[dp] = {
                "job_id": jid,
                "detector_preset": dp,
                "yolo_weights_sha256": (res.get("composite_metadata") or {}).get("yolo_weights_sha256"),
                "official_025_count": len(res.get("predictions") or []),
                "low_conf_pool_005_count": (res.get("composite_metadata") or {}).get("low_conf_pool_005_count"),
                "client_poll_wall_seconds": run["wall_seconds"],
                "artifacts_dir": str(a_dir),
            }

        # Run composite natural-language / spatial / absent-class queries on the FICS R15 job
        fics_jid = arms_out["fics_yolo11n_r15"]["job_id"]
        query_records: List[Dict[str, Any]] = []
        for q_idx, q_text in enumerate(bspec["queries"]):
            q_out = _api_json(
                f"{base_url}/api/composite/query",
                method="POST",
                body={"job_id": fics_jid, "query_text": q_text},
            )
            q_file = b_dir / "fics_yolo11n_r15" / f"query_{q_idx:02d}.json"
            _write_and_lock_json(q_file, q_out, f"{bkey}_fics_query_{q_idx:02d}")
            query_records.append({
                "query_index": q_idx,
                "query_id": q_out["query_id"],
                "query_text": q_text,
                "predicted_count": q_out["predicted_count"],
                "query_file": str(q_file),
            })

        board_records[bkey] = {
            "board_key": bkey,
            "source_tif_path": str(tif_p),
            "source_tif_sha256": _sha256_file(tif_p),
            "upload_id": uid,
            "orig_wh": up["orig_wh"],
            "roi_bbox_half_open": up["roi_bbox_half_open"],
            "arms": arms_out,
            "fics_queries": query_records,
        }

    lock_doc = {
        "schema_version": "r15_pre_gt_lock_v1",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "web_inference_total_wall_seconds": round(time.perf_counter() - t_all0, 3),
        "gt_loaded_during_web_inference": False,
        "reserved_blind_boards_untouched": ["s31", "s10", "s16", "s19"],
        "excluded_overlap_untouched": ["s9"],
        "boards": board_records,
        "locked_files": locked_files,
    }
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.lock_file.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"locked_files_count": len(locked_files), "web_inference_total_wall_seconds": lock_doc["web_inference_total_wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
