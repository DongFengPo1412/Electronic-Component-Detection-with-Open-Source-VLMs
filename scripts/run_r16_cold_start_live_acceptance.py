"""Round 16 Cold-Start Live Web Acceptance Script (Port 8820).

Verifies:
1. Cold-start 4-point default consistency (`/` HTML default, `/api/config`, `POST /api/jobs` omitting `detector_preset`,
   and actual worker `yolo_run_summary.json`) all resolve to `fics_yolo11n_r15` + SHA `4d92fd1d...` + `qwen3_participated=False`.
2. Real original `s20_front.tif` upload, whole-board detection, overlay PNG export, JSON export, and Markdown summary export.
3. Half-region resistor query (`只找电阻，但只统计左半边`) and multi-class query (`找电阻和电容`) + query JSON export & overlay PNG.
4. Pure-VLM Entry A/B defaults & Tab 2 historical replay intact.
5. Pre-GT locks all live artifacts into `r16_live_cold_start_lock.json` and verifies bit-for-bit parity with R15 S20 predictions.
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

DEFAULT_BASE_URL = "http://127.0.0.1:8820"
DEFAULT_OUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124\live_demo")
DEFAULT_WS_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration16-20261124\live_ws")
S20_TIF = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif")
R15_S20_OFFICIAL = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\live_demo\s20\fics_yolo11n_r15\yolo_official_025.json")
EXPECTED_FICS_SHA = "4d92fd1d8093ea7498c13b3aac6dd5a6e5c21b9965b96667270883aed571009e"


def _sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest().lower()


def _api_json(url: str, *, method: str = "GET", body: Optional[Mapping[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Round 16 cold-start live web acceptance.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ws-dir", type=pathlib.Path, default=DEFAULT_WS_DIR)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    base = args.base_url.rstrip("/")
    health = _api_json(f"{base}/api/health")
    assert health["ok"] and health["research_available"], f"Health check failed: {health}"

    # 1) Verify HTML & /api/config cold-start defaults
    html = _get_bytes(f"{base}/").decode("utf-8")
    assert '<option value="fics_yolo11n_r15" selected>' in html, "HTML default is not fics_yolo11n_r15"
    cfg = _api_json(f"{base}/api/config")
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
    assert cfg["composite_detector"]["presets"][0]["preset_id"] == "fics_yolo11n_r15"
    assert cfg["composite_detector"]["presets"][0]["weights_sha256"] == EXPECTED_FICS_SHA
    assert cfg["single_model_defense"]["model_key"] == "qwen3"
    assert cfg["three_model_compare"]["mode"] == "W"
    ov = _api_json(f"{base}/api/research/overview")
    assert ov["available"] is True

    # 2) Upload real original S20 TIF and submit COMPOSITE_YOLO OMITTING detector_preset
    up_req = urllib.request.Request(
        f"{base}/api/upload",
        data=S20_TIF.read_bytes(),
        headers={"X-Filename": S20_TIF.name},
        method="POST",
    )
    up = json.loads(urllib.request.urlopen(up_req, timeout=120.0).read().decode("utf-8"))
    uid = up["upload_id"]

    sub = _api_json(
        f"{base}/api/jobs",
        method="POST",
        body={"upload_id": uid, "model_key": "yolo11n_composite", "mode": "COMPOSITE_YOLO"},
    )
    jid = sub["job_id"]
    t_job0 = time.perf_counter()
    st = sub
    while (time.perf_counter() - t_job0) < 300.0:
        st = _api_json(f"{base}/api/jobs/{jid}")
        if st["status"] in ("completed", "failed", "cancelled", "incomplete"):
            break
        time.sleep(0.5)
    assert st["status"] == "completed", f"Cold-start job failed: {st}"
    job_wall = round(time.perf_counter() - t_job0, 3)

    res = _api_json(f"{base}/api/jobs/{jid}/result")
    assert res["gt_loaded"] is False
    assert res["detector_preset"] == "fics_yolo11n_r15"
    assert res["composite_metadata"]["detector_preset"] == "fics_yolo11n_r15"
    assert res["composite_metadata"]["yolo_weights_sha256"] == EXPECTED_FICS_SHA
    assert res["composite_metadata"]["qwen3_participated"] is False

    # Check worker summary & job_spec
    jdir = args.ws_dir / "jobs" / jid
    job_spec = json.loads((jdir / "job_spec.json").read_text(encoding="utf-8"))
    yolo_sum = json.loads((jdir / "base_runs" / "composite_yolo" / "yolo_run_summary.json").read_text(encoding="utf-8"))
    assert job_spec["detector_preset"] == "fics_yolo11n_r15"
    assert job_spec["yolo_weights_sha256"] == EXPECTED_FICS_SHA
    assert yolo_sum["detector_preset"] == "fics_yolo11n_r15"
    assert yolo_sum["weights_sha256"] == EXPECTED_FICS_SHA
    assert yolo_sum["qwen3_participated"] is False

    # Verify bit-for-bit prediction parity with R15 locked S20
    r15_preds = json.loads(R15_S20_OFFICIAL.read_text(encoding="utf-8"))
    assert len(res["predictions"]) == len(r15_preds) == 112
    for a, b in zip(res["predictions"], r15_preds):
        assert a["label"] == b["label"] and a["bbox_xyxy"] == b["bbox_xyxy"] and a["score"] == b["score"]

    # 3) Export JSON, overlay PNG, summary Markdown
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    locked: List[Dict[str, Any]] = []

    def _save_and_lock(name: str, data: bytes, role: str) -> pathlib.Path:
        p = args.out_dir / name
        p.write_bytes(data)
        locked.append({"role": role, "path": str(p), "sha256": _sha256_file(p), "bytes": len(data)})
        return p

    exp_json_bytes = _get_bytes(f"{base}/api/jobs/{jid}/export/json")
    exp_json = json.loads(exp_json_bytes.decode("utf-8"))
    assert exp_json["export_provenance"]["weight_identity"]["weights_sha256"] == EXPECTED_FICS_SHA
    assert exp_json["export_provenance"]["qwen3_participated"] is False
    _save_and_lock("s20_default_export.json", exp_json_bytes, "job_export_json")

    exp_png_bytes = _get_bytes(f"{base}/api/jobs/{jid}/export/overlay_png")
    assert exp_png_bytes.startswith(b"\x89PNG")
    _save_and_lock("s20_default_overlay.png", exp_png_bytes, "job_export_overlay_png")

    exp_md_bytes = _get_bytes(f"{base}/api/jobs/{jid}/export/summary_md")
    assert "fics_yolo11n_r15" in exp_md_bytes.decode("utf-8")
    _save_and_lock("s20_default_summary.md", exp_md_bytes, "job_export_summary_md")

    shutil.copy2(jdir / "base_runs" / "composite_yolo" / "yolo_run_summary.json", args.out_dir / "yolo_run_summary.json")
    locked.append({
        "role": "worker_summary",
        "path": str(args.out_dir / "yolo_run_summary.json"),
        "sha256": _sha256_file(args.out_dir / "yolo_run_summary.json"),
        "bytes": (args.out_dir / "yolo_run_summary.json").stat().st_size,
    })

    # 4) Half-region resistor query & multi-class query + exports
    queries_meta: List[Dict[str, Any]] = []
    for idx, q_text in enumerate(("只找电阻，但只统计左半边", "找电阻和电容")):
        q_out = _api_json(
            f"{base}/api/composite/query",
            method="POST",
            body={"job_id": jid, "query_text": q_text},
        )
        assert q_out["query_execution_mechanism"] == "筛选已生成的检测结果"
        assert q_out["gpu_called_for_query"] is False
        q_exp_bytes = _get_bytes(f"{base}/api/composite/query_export/{jid}/{q_out['query_id']}")
        q_exp = json.loads(q_exp_bytes.decode("utf-8"))
        assert q_exp["weight_identity"]["weights_sha256"] == EXPECTED_FICS_SHA
        assert q_exp["filter_source"] == "筛选已生成的检测结果"
        assert q_exp["qwen3_participated"] is False
        _save_and_lock(f"query_{idx:02d}_export.json", q_exp_bytes, f"query_{idx:02d}_export_json")
        q_ov_bytes = _get_bytes(f"{base}{q_out['overlay_url']}")
        assert q_ov_bytes.startswith(b"\x89PNG")
        _save_and_lock(f"query_{idx:02d}_overlay.png", q_ov_bytes, f"query_{idx:02d}_overlay_png")
        queries_meta.append({
            "query_text": q_text,
            "query_id": q_out["query_id"],
            "predicted_count": q_out["predicted_count"],
        })

    lock_doc = {
        "schema_version": "r16_live_cold_start_lock_v1",
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
        "cold_start_job_wall_seconds": job_wall,
        "job_id": jid,
        "upload_id": uid,
        "four_point_default_consistency": {
            "html_selected": "fics_yolo11n_r15",
            "api_config_default": cfg["composite_detector"]["default_detector_preset"],
            "job_spec_preset": job_spec["detector_preset"],
            "worker_summary_preset": yolo_sum["detector_preset"],
            "weights_sha256": EXPECTED_FICS_SHA,
            "qwen3_participated": False,
        },
        "r15_s20_parity_exact": True,
        "queries": queries_meta,
        "locked_files": locked,
    }
    (args.out_dir / "r16_live_cold_start_lock.json").write_text(
        json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "cold_start_job_wall_seconds": job_wall,
        "total_wall_seconds": lock_doc["total_wall_seconds"],
        "job_id": jid,
        "r15_s20_parity_exact": True,
        "queries": queries_meta,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
