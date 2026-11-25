"""Pure GT-free Round 17 real web upload, paired inference, language queries & pre-GT lock script."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8821"
DEFAULT_OUTPUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\live_demo")
DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\eval\r17_pre_gt_lock.json")
DEFAULT_WORKSPACE_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration17-20261124\live_ws")

BOARDS_SPEC: List[Dict[str, Any]] = [
    {
        "board_key": "s20",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif",
        "queries": [
            "只找电阻，但只统计左半边",
            "只找二极管并计数",
            "只找IC并计数",
        ],
    },
    {
        "board_key": "s4",
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif",
        "queries": [
            "只找晶体管并计数",
            "只找IC并计数",
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
    detector_preset: Optional[str],
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "upload_id": upload_id,
        "model_key": "yolo11n_composite",
        "mode": "COMPOSITE_YOLO",
    }
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
        raise RuntimeError(f"Composite job {jid} ({detector_preset}) failed: {st}")
    res = _api_json(f"{base_url}/api/jobs/{jid}/result")
    assert res["gt_loaded"] is False, f"GT must NOT be loaded during live inference! {jid}"
    return {"status": st, "result": res, "wall_seconds": round(time.perf_counter() - t0, 3)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run GT-free paired composite web inference for R17 and lock predictions.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lock-file", type=pathlib.Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--workspace-dir", type=pathlib.Path, default=DEFAULT_WORKSPACE_DIR)
    args = parser.parse_args(argv)

    t_all0 = time.perf_counter()
    base_url = args.base_url.rstrip("/")
    health = _api_json(f"{base_url}/api/health")
    assert health.get("ok"), f"Health check failed: {health}"

    html = urllib.request.urlopen(f"{base_url}/", timeout=15.0).read().decode("utf-8")
    assert '<option value="fics_yolo11n_r15" selected>' in html, "HTML default option must remain fics_yolo11n_r15"
    cfg = _api_json(f"{base_url}/api/config")
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
    preset_ids = [p["preset_id"] for p in cfg["composite_detector"]["presets"]]
    assert "fics_yolo11n_r17_cand" in preset_ids

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

    def _download_and_lock(url: str, dst: pathlib.Path, role: str) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(url, timeout=30.0).read())
        sha = _sha256_file(dst)
        locked_files.append({"role": role, "path": str(dst), "sha256": sha, "bytes": dst.stat().st_size})
        return sha

    for bspec in BOARDS_SPEC:
        bkey = str(bspec["board_key"])
        tif_p = pathlib.Path(str(bspec["tif_path"]))
        assert bkey not in ("s12", "s24", "s17", "s31", "s10", "s16", "s19", "s9"), f"Banned board: {bkey}"
        up = _upload_tif(base_url, tif_p)
        uid = str(up["upload_id"])
        b_dir = args.output_dir / bkey
        _write_and_lock_json(b_dir / "upload_meta.json", up, f"{bkey}_upload_meta")

        arms_out: Dict[str, Any] = {}
        arm_specs = [
            ("fics_yolo11n_r15", None if bkey == "s20" else "fics_yolo11n_r15"),
            ("fics_yolo11n_r17_cand", "fics_yolo11n_r17_cand"),
        ]
        for arm_name, submit_dp in arm_specs:
            run = _run_and_poll_composite(base_url=base_url, upload_id=uid, detector_preset=submit_dp)
            res = run["result"]
            jid = str(res["job_id"])
            assert res.get("detector_preset") == arm_name, f"Preset mismatch on {bkey}: {res.get('detector_preset')}"
            assert (res.get("composite_metadata") or {}).get("detector_preset") == arm_name

            j_ws_dir = args.workspace_dir / "jobs" / jid
            comp_ws_dir = j_ws_dir / "base_runs" / "composite_yolo"
            a_dir = b_dir / arm_name

            _write_and_lock_json(a_dir / "job_result.json", res, f"{bkey}_{arm_name}_job_result")
            _copy_and_lock(j_ws_dir / "job_spec.json", a_dir / "job_spec.json", f"{bkey}_{arm_name}_job_spec")
            _copy_and_lock(comp_ws_dir / "yolo_run_summary.json", a_dir / "yolo_run_summary.json", f"{bkey}_{arm_name}_summary")
            _copy_and_lock(comp_ws_dir / "yolo_official_025.json", a_dir / "yolo_official_025.json", f"{bkey}_{arm_name}_preds_025")
            _copy_and_lock(comp_ws_dir / "yolo_low_conf_pool_005.json", a_dir / "yolo_low_conf_pool_005.json", f"{bkey}_{arm_name}_pool_005")
            _copy_and_lock(comp_ws_dir / "yolo_raw_outputs_all.jsonl", a_dir / "yolo_raw_outputs_all.jsonl", f"{bkey}_{arm_name}_raw_calls")
            if res.get("overlay_url"):
                _download_and_lock(f"{base_url}{res['overlay_url']}", a_dir / "overlay.png", f"{bkey}_{arm_name}_overlay")
            _download_and_lock(f"{base_url}/api/jobs/{jid}/export/json", a_dir / "export.json", f"{bkey}_{arm_name}_export")

            arms_out[arm_name] = {
                "job_id": jid,
                "omitted_detector_preset": submit_dp is None,
                "client_poll_wall_seconds": run["wall_seconds"],
                "artifacts_dir": str(a_dir),
            }

        cand_jid = arms_out["fics_yolo11n_r17_cand"]["job_id"]
        q_records: List[Dict[str, Any]] = []
        for q_idx, q_text in enumerate(bspec["queries"]):
            q_res = _api_json(
                f"{base_url}/api/composite/query",
                method="POST",
                body={"job_id": cand_jid, "query_text": q_text},
            )
            qid = str(q_res["query_id"])
            q_json_dst = b_dir / f"r17_cand_query_{q_idx:02d}.json"
            _write_and_lock_json(q_json_dst, q_res, f"{bkey}_query_{q_idx:02d}_json")
            q_ov_dst = b_dir / f"r17_cand_query_{q_idx:02d}_overlay.png"
            if q_res.get("overlay_url"):
                _download_and_lock(f"{base_url}{q_res['overlay_url']}", q_ov_dst, f"{bkey}_query_{q_idx:02d}_overlay")
            q_records.append({
                "query_id": qid,
                "raw_query": q_text,
                "predicted_count": q_res["predicted_count"],
                "query_file": str(q_json_dst),
                "overlay_file": str(q_ov_dst),
            })

        board_records[bkey] = {
            "upload_id": uid,
            "tif_path": str(tif_p),
            "tif_sha256": _sha256_file(tif_p),
            "roi_bbox_half_open": up["roi_bbox_half_open"],
            "arms": arms_out,
            "r17_cand_queries": q_records,
        }
        print(f"[LOCKED] {bkey}: r15={arms_out['fics_yolo11n_r15']['job_id']}, r17_cand={cand_jid}, queries={len(q_records)}", flush=True)

    lock_doc = {
        "schema_version": "r17_pre_gt_lock_v1",
        "base_url": base_url,
        "total_wall_seconds": round(time.perf_counter() - t_all0, 3),
        "cold_start_default_verified_on_s20": True,
        "gt_loaded_during_inference": False,
        "boards": board_records,
        "locked_files": locked_files,
    }
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.lock_file.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE pre-GT lock: {len(locked_files)} files locked -> {args.lock_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
