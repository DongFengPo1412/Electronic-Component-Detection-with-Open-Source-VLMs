"""Recover official FICS-PCB DSLR boards via chunked HTTP 206 Range, audit families/FPIC, and build Codex R17 blind package.

Strict invariants:
- Zero GPU inference and zero image preview on blind boards (`s12_front`, `s24_front`).
- Supports 32-bit wrapped ZIP Central Directories (>4 GiB FICS archives) with CRC32 & uncompressed-size verification.
- Logs all HTTP bytes in `data_audit/r17_network_ledger.json`.
"""

from __future__ import annotations

import collections
import concurrent.futures
import csv
import hashlib
import http.client
import io
import json
import pathlib
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

socket.setdefaulttimeout(25.0)

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import (  # noqa: E402
    FICS_CANONICAL_CATEGORIES,
    FICS_CATEGORY_NAME_TO_ID,
    FICS_RAW_TO_CANONICAL,
    file_sha256,
    parse_component_location,
    validate_box_within_image,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402

R17_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration17-20261124")
R17_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124")
FETCH_ROOT = R17_CACHE / "fetched_fics_boards"
AUDIT_DIR = R17_ART / "data_audit"
BLIND_DIR = R17_ART / "codex_reserved_blind"
LEDGER_PATH = AUDIT_DIR / "r17_network_ledger.json"

R14_INVENTORY_PATH = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json"
)
FPIC_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-dataset-alignment-20261114\fpic_official")

BASE_URL = "https://www.trust-hub.org/downloads/resource/data/fics-pcb/"
CHUNK_BYTES = 4 * 1024 * 1024
MAX_RETRIES = 6

FETCH_SPECS: List[Dict[str, Any]] = [
    {
        "board_id": "s14",
        "role": "train_expansion",
        "physical_family": "mitel_tms320_lattice_telecom",
        "family_proof": "Mitel MT90863AG + TI TMS320VC549PGE + Lattice ispLSI 2064VE (grouped with sister s15 in train_pool)",
        "sides": ["s14_front1", "s14_front2", "s14_back1", "s14_back2"],
    },
    {
        "board_id": "s15",
        "role": "train_expansion",
        "physical_family": "mitel_tms320_lattice_telecom",
        "family_proof": "Mitel MT90863AG + TI TMS320VC549PGE + Lattice ispLSI 2064VE (grouped with sister s14 in train_pool)",
        "sides": ["s15_front1", "s15_front2", "s15_back1", "s15_back2"],
    },
    {
        "board_id": "s26",
        "role": "train_expansion",
        "physical_family": "renesas_h8_3026_lm358_controller",
        "family_proof": "Renesas H8/3026 + LM358A + 74LV125A + 24C32WP (all 4 views grouped in train_pool)",
        "sides": ["s26_front1", "s26_front2", "s26_back1", "s26_back2"],
    },
    {
        "board_id": "s25",
        "role": "train_expansion",
        "physical_family": "hitachi_lsi_sas_hdd",
        "family_proof": "Hitachi 0B23571 + LSI 0B24025 + Samsung K4T56163Q0",
        "sides": ["s25_front"],
    },
    {
        "board_id": "s23",
        "role": "train_expansion",
        "physical_family": "ibm36_amsrc04_hdd",
        "family_proof": "IBM36 JAPAN AMSRC04 + TDK (same IBM HDD family as R15 train_pool s21; zero val/dev/blind overlap)",
        "sides": ["s23_front"],
    },
    {
        "board_id": "s12",
        "role": "codex_blind_weak_positive",
        "physical_family": "phl723_78664_highres",
        "family_proof": "78664;S23786;PHL 723 + 0.68UH;1645A + 125.000;Nd629A (unseen weak-class-positive blind board)",
        "sides": ["s12_front"],
    },
    {
        "board_id": "s24",
        "role": "codex_blind_ic_absent_weak",
        "physical_family": "at49f1024_winbond_w981616",
        "family_proof": "AT49F1024;45VC;0252 + Winbond W981616BH-6 (unseen IC + absent diode/transistor blind board)",
        "sides": ["s24_front"],
    },
]


class Ledger:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self.total_bytes = 0
        self._lock = threading.Lock()

    def record(self, ev: Dict[str, Any]) -> None:
        n = int(ev.get("bytes", 0))
        with self._lock:
            self.total_bytes += n
            ev["cumulative_bytes"] = self.total_bytes
            self.events.append(ev)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": "r17_network_ledger_v1",
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "cumulative_http_bytes_read": self.total_bytes,
                    "event_count": len(self.events),
                    "events": self.events,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def _head(opener: urllib.request.OpenerDirector, url: str, bid: str, ledger: Ledger) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
            )
            with opener.open(req, timeout=30.0) as r:
                cl = int(r.headers.get("Content-Length") or 0)
                et = (r.headers.get("ETag") or "").strip()
                ar = (r.headers.get("Accept-Ranges") or "").strip()
                ledger.record({"board_id": bid, "method": "HEAD", "status": int(r.status), "bytes": 0, "etag": et})
                return {"length": cl, "etag": et, "accept_ranges": ar}
        except Exception as exc:
            if attempt + 1 >= MAX_RETRIES:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _range_get(
    opener: urllib.request.OpenerDirector,
    url: str,
    bid: str,
    start: int,
    end: int,
    total_len: int,
    etag: str,
    purpose: str,
    ledger: Ledger,
) -> bytes:
    exp_len = end - start + 1
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
            }
            if etag:
                headers["If-Range"] = etag
            req = urllib.request.Request(url, method="GET", headers=headers)
            with opener.open(req, timeout=30.0) as r:
                if int(r.status) != 206:
                    raise RuntimeError(f"Expected 206, got {r.status} for {bid} {purpose}")
                parts: List[bytes] = []
                got = 0
                while got < exp_len:
                    blk = r.read(min(262144, exp_len - got))
                    if not blk:
                        break
                    parts.append(blk)
                    got += len(blk)
                data = b"".join(parts)
                if len(data) != exp_len:
                    raise RuntimeError(f"Short read {len(data)} != {exp_len} for {bid} {purpose}")
                ledger.record(
                    {
                        "board_id": bid,
                        "method": "GET_RANGE",
                        "purpose": purpose,
                        "range": f"{start}-{end}",
                        "bytes": len(data),
                        "attempt": attempt + 1,
                    }
                )
                return data
        except Exception as exc:
            ledger.record(
                {
                    "board_id": bid,
                    "method": "GET_RANGE_ERR",
                    "purpose": purpose,
                    "range": f"{start}-{end}",
                    "bytes": 0,
                    "attempt": attempt + 1,
                    "error": repr(exc),
                }
            )
            if attempt + 1 >= MAX_RETRIES:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _parse_cd(cd_bytes: bytes) -> List[Dict[str, Any]]:
    pos = 0
    out: List[Dict[str, Any]] = []
    n = len(cd_bytes)
    while pos + 46 <= n and cd_bytes[pos : pos + 4] == b"PK\x01\x02":
        (_, _, _, _, comp, _, _, crc, csz, usz, fnlen, exlen, cmlen, _, _, _, lh_off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd_bytes[pos : pos + 46]
        )
        fn = cd_bytes[pos + 46 : pos + 46 + fnlen].decode("utf-8", errors="replace").replace("\\", "/")
        extra = cd_bytes[pos + 46 + fnlen : pos + 46 + fnlen + exlen]
        ep = 0
        while ep + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[ep : ep + 4])
            if hid == 0x0001:
                zd = extra[ep + 4 : ep + 4 + hsz]
                zp = 0
                if usz == 0xFFFFFFFF and zp + 8 <= len(zd):
                    usz = struct.unpack("<Q", zd[zp : zp + 8])[0]
                    zp += 8
                if csz == 0xFFFFFFFF and zp + 8 <= len(zd):
                    csz = struct.unpack("<Q", zd[zp : zp + 8])[0]
                    zp += 8
                if lh_off == 0xFFFFFFFF and zp + 8 <= len(zd):
                    lh_off = struct.unpack("<Q", zd[zp : zp + 8])[0]
                    zp += 8
            ep += 4 + hsz
        out.append({"fn": fn, "comp": comp, "crc": crc, "csz": csz, "usz": usz, "lh_off": lh_off})
        pos += 46 + fnlen + exlen + cmlen
    return out


def _read_zip_catalog(
    opener: urllib.request.OpenerDirector, url: str, bid: str, total_len: int, etag: str, ledger: Ledger
) -> List[Dict[str, Any]]:
    tail_size = min(total_len, 65558)
    tail_start = total_len - tail_size
    tail = _range_get(opener, url, bid, tail_start, total_len - 1, total_len, etag, "eocd_tail", ledger)
    p = tail.rfind(b"PK\x05\x06")
    if p < 0:
        raise RuntimeError(f"EOCD not found in {bid}")
    eocd_abs = tail_start + p
    _, _, _, _, _, cd_sz, cd_off, _ = struct.unpack("<IHHHHIIH", tail[p : p + 22])
    real_cd_off = eocd_abs - cd_sz
    lp = tail.rfind(b"PK\x06\x07", 0, p)
    if lp >= 0:
        _, _, z64_off, _ = struct.unpack("<IIQI", tail[lp : lp + 20])
        z64 = _range_get(opener, url, bid, z64_off, z64_off + 55, total_len, etag, "zip64_eocd", ledger)
        _, _, _, _, _, _, _, _, cd_sz, real_cd_off = struct.unpack("<IQHHIIQQQQ", z64[:56])
    cd_bytes = _range_get(opener, url, bid, real_cd_off, real_cd_off + cd_sz - 1, total_len, etag, "central_dir", ledger)
    if cd_bytes[:4] != b"PK\x01\x02":
        raise RuntimeError(f"Invalid CD magic for {bid}: {cd_bytes[:4]!r}")
    return _parse_cd(cd_bytes)


def _extract_entry_to_file(
    opener: urllib.request.OpenerDirector,
    url: str,
    bid: str,
    total_len: int,
    etag: str,
    entry: Dict[str, Any],
    dest: pathlib.Path,
    ledger: Ledger,
) -> Dict[str, Any]:
    if dest.is_file() and dest.stat().st_size == int(entry["usz"]):
        existing_crc = zlib.crc32(dest.read_bytes()) & 0xFFFFFFFF
        if existing_crc == int(entry["crc"]):
            return {
                "path": str(dest),
                "sha256": file_sha256(dest).lower(),
                "bytes": dest.stat().st_size,
                "crc32_hex": f"{existing_crc:08x}",
                "cached": True,
            }

    # Resolve true local file header offset (handles >4 GiB 32-bit wrapped offsets)
    lh_abs = None
    data_start = None
    for k in range(4):
        cand_off = int(entry["lh_off"]) + k * (1 << 32)
        if cand_off + 30 >= total_len:
            break
        hdr = _range_get(
            opener, url, bid, cand_off, min(total_len - 1, cand_off + 511), total_len, etag, f"lh_{dest.name}_k{k}", ledger
        )
        if hdr[:4] == b"PK\x03\x04":
            _, _, _, comp, _, _, _, _, _, fnl, exl = struct.unpack("<IHHHHHIIIHH", hdr[:30])
            fn_loc = hdr[30 : 30 + fnl].decode("utf-8", errors="replace").replace("\\", "/")
            if fn_loc == entry["fn"]:
                lh_abs = cand_off
                data_start = cand_off + 30 + fnl + exl
                break
    if lh_abs is None or data_start is None:
        raise RuntimeError(f"Local header not found for {entry['fn']} in {bid}")

    comp_size = int(entry["csz"])
    comp_type = int(entry["comp"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    decomp = zlib.decompressobj(-15) if comp_type == 8 else None
    crc_acc = 0
    written = 0
    ranges: List[Tuple[int, int, int]] = []
    pos = 0
    while pos < comp_size:
        take = min(CHUNK_BYTES, comp_size - pos)
        ranges.append((pos, data_start + pos, data_start + pos + take - 1))
        pos += take

    def _fetch_one(item: Tuple[int, int, int]) -> bytes:
        off, s, e = item
        w_opener = _opener()
        return _range_get(
            w_opener, url, bid, s, e, total_len, etag, f"dl_{dest.name}_{off}_{off + (e - s + 1)}", ledger
        )

    with tmp_dest.open("wb") as out_f:
        if len(ranges) <= 1:
            for item in ranges:
                chunk = _fetch_one(item)
                plain = decomp.decompress(chunk) if decomp is not None else chunk
                if plain:
                    out_f.write(plain)
                    out_f.flush()
                    crc_acc = zlib.crc32(plain, crc_acc)
                    written += len(plain)
                done_b = item[0] + (item[2] - item[1] + 1)
                print(f"  [DL] {dest.name}: {done_b}/{comp_size} ({100.0*done_b/comp_size:.1f}%)", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for win_start in range(0, len(ranges), 8):
                    win = ranges[win_start : win_start + 8]
                    futs = [pool.submit(_fetch_one, it) for it in win]
                    for it, fut in zip(win, futs):
                        chunk = fut.result()
                        plain = decomp.decompress(chunk) if decomp is not None else chunk
                        if plain:
                            out_f.write(plain)
                            out_f.flush()
                            crc_acc = zlib.crc32(plain, crc_acc)
                            written += len(plain)
                        done_b = it[0] + (it[2] - it[1] + 1)
                        print(f"  [DL] {dest.name}: {done_b}/{comp_size} ({100.0*done_b/comp_size:.1f}%)", flush=True)
        if decomp is not None:
            tail_plain = decomp.flush()
            if tail_plain:
                out_f.write(tail_plain)
                crc_acc = zlib.crc32(tail_plain, crc_acc)
                written += len(tail_plain)

    crc_acc &= 0xFFFFFFFF
    if crc_acc != int(entry["crc"]) or written != int(entry["usz"]):
        tmp_dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"CRC/size mismatch for {dest.name}: crc={crc_acc:08x}/{int(entry['crc']):08x}, size={written}/{entry['usz']}"
        )
    if dest.exists():
        dest.unlink()
    tmp_dest.rename(dest)
    return {
        "path": str(dest),
        "sha256": file_sha256(dest).lower(),
        "bytes": written,
        "crc32_hex": f"{crc_acc:08x}",
        "cached": False,
    }


def _decoded_rgb_sha256(img_path: pathlib.Path) -> Tuple[str, Tuple[int, int]]:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        h_obj = hashlib.sha256()
        h_obj.update(f"{w}x{h}:".encode("ascii"))
        h_obj.update(rgb.tobytes())
        return h_obj.hexdigest(), (w, h)


def _validate_csv(csv_path: pathlib.Path, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    raw_to_canon_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    gts: List[Dict[str, Any]] = []
    cls_counts: collections.Counter[str] = collections.Counter()
    dropped: collections.Counter[str] = collections.Counter()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row_num, row in enumerate(csv.DictReader(fh), start=2):
            raw_type = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(raw_type) or raw_to_canon_lower.get(raw_type.lower())
            if canon is None:
                dropped[raw_type or "<EMPTY>"] += 1
                continue
            _raw, bbox_xyxy, note = parse_component_location(row.get("component_location") or "")
            validate_box_within_image(bbox_xyxy, img_w, img_h)
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            gts.append(
                {
                    "gt_index": len(gts),
                    "csv_row": row_num,
                    "label": canon,
                    "coco_category_id": FICS_CATEGORY_NAME_TO_ID[canon],
                    "raw_component_type": raw_type,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "area_px2": (x2 - x1) * (y2 - y1),
                    "conversion_note": note,
                }
            )
            cls_counts[canon] += 1
    return gts, {c: int(cls_counts.get(c, 0)) for c in FICS_SIX_CLASSES}, dict(dropped)


def _write_coco(
    *,
    stem: str,
    board_id: str,
    side: str,
    tif_path: pathlib.Path,
    tif_sha: str,
    csv_path: pathlib.Path,
    csv_sha: str,
    gts: Sequence[Mapping[str, Any]],
    img_w: int,
    img_h: int,
    out_path: pathlib.Path,
) -> Dict[str, Any]:
    coco = {
        "info": {
            "dataset": "FICS-PCB_DSLR_Official_Single_Board",
            "version": "r17_fics6_v1",
            "board_side_id": stem,
            "source_csv": str(csv_path),
            "source_csv_sha256": csv_sha,
        },
        "images": [
            {
                "id": 1,
                "sample_id": stem,
                "file_name": tif_path.name,
                "width": int(img_w),
                "height": int(img_h),
                "board_id": board_id,
                "side": side,
                "sha256": tif_sha,
            }
        ],
        "categories": list(FICS_CANONICAL_CATEGORIES),
        "annotations": [
            {
                "id": idx + 1,
                "image_id": 1,
                "category_id": int(g["coco_category_id"]),
                "bbox": [
                    round(float(g["bbox_xyxy"][0]), 4),
                    round(float(g["bbox_xyxy"][1]), 4),
                    round(float(g["bbox_xyxy"][2] - g["bbox_xyxy"][0]), 4),
                    round(float(g["bbox_xyxy"][3] - g["bbox_xyxy"][1]), 4),
                ],
                "bbox_xyxy": [round(float(v), 4) for v in g["bbox_xyxy"]],
                "area": round(float(g["area_px2"]), 4),
                "iscrowd": 0,
                "raw_component_type": str(g["raw_component_type"]),
                "csv_row": int(g["csv_row"]),
            }
            for idx, g in enumerate(gts)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(out_path), "sha256": file_sha256(out_path).lower(), "annotation_count": len(coco["annotations"])}


def _audit_local_fpic() -> Dict[str, Any]:
    pcb_csv = FPIC_DIR / "metadata" / "metadata" / "pcb.csv"
    smd_zip = FPIC_DIR / "smd_annotation.zip"
    if not smd_zip.is_file() or not pcb_csv.is_file():
        return {"present": False}
    pcb_rows: Dict[str, Dict[str, str]] = {}
    with pcb_csv.open("r", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            pcb_rows[r["id"].strip()] = r
    total_objs = 0
    empty_des = 0
    des_pref: collections.Counter[str] = collections.Counter()
    with zipfile.ZipFile(smd_zip, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            raw = zf.read(name).decode("utf-8", errors="replace")
            for r in csv.DictReader(io.StringIO(raw)):
                total_objs += 1
                des = (r.get("Designator") or "").strip().upper()
                if not des:
                    empty_des += 1
                else:
                    m = "".join(ch for ch in des if ch.isalpha())[:2]
                    for idx, ch in enumerate(des):
                        if not ch.isalpha():
                            m = des[:idx]
                            break
                    else:
                        m = des
                    des_pref[m] += 1
    return {
        "present": True,
        "source_url": "https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/",
        "smd_annotation_zip_sha256": file_sha256(smd_zip).lower(),
        "total_smd_rows": total_objs,
        "empty_designator_rows": empty_des,
        "candidate_prefix_counts": {k: int(des_pref[k]) for k in ("D", "CR", "Q", "U", "IC", "R", "C", "L", "LED")},
        "known_fpic_fics_physical_overlaps": [
            "pcb_2 = s10 (HannStar blue VGA)",
            "pcb_3 = s20 (TI MSP-EXP430G2ET)",
            "pcb_4 = s17 (TI MSP-EXP430FR2433)",
            "pcb_7 = s9 (Digilent Zybo Zynq-7000)",
            "pcb_39 = s16 (Numato Mimas V2)",
            "pcb_9 / pcb_32 / pcb_34 = Cisco WIC family (s2/s3/s1)",
        ],
        "non_overlapping_fpic_fallback_candidates": ["pcb_91", "pcb_92", "pcb_31", "pcb_49", "pcb_8"],
        "r17_decision": "Not mixed into R17 because pure official FICS-PCB DSLR recovery (`s14/s15/s26/s25/s23` + `s12/s24`) already satisfies >=2 new families and >=10 verified instances per weak class without designator/unlabeled-background ambiguity.",
    }


def main() -> int:
    opener = _opener()
    ledger = Ledger(LEDGER_PATH)

    # Check pixel SHA against all existing R14/R15 boards (val, dev, sealed)
    r14_inv = json.loads(R14_INVENTORY_PATH.read_text(encoding="utf-8"))
    existing_pixel_shas: Dict[str, str] = {}
    for b in r14_inv["boards"]:
        tp = pathlib.Path(str(b["tif_path"]))
        if tp.is_file():
            px, _ = _decoded_rgb_sha256(tp)
            existing_pixel_shas[px] = str(b["board_side_id"])

    recovered_sides: List[Dict[str, Any]] = []
    blind_records: List[Dict[str, Any]] = []

    for spec in FETCH_SPECS:
        bid = str(spec["board_id"])
        role = str(spec["role"])
        url = f"{BASE_URL}{bid}.zip"
        hmeta = _head(opener, url, bid, ledger)
        total_len = int(hmeta["length"])
        etag = str(hmeta["etag"])
        entries = _read_zip_catalog(opener, url, bid, total_len, etag, ledger)
        by_base = {pathlib.PurePosixPath(e["fn"]).name: e for e in entries if "/DSLR/" in e["fn"] and "__MACOSX" not in e["fn"] and not e["fn"].endswith("/")}

        base_dest = (BLIND_DIR / bid) if role.startswith("codex_blind") else (FETCH_ROOT / bid)
        for side_stem in spec["sides"]:
            tif_name = f"{side_stem}.tif"
            csv_name = f"{side_stem}.csv"
            if tif_name not in by_base or csv_name not in by_base:
                raise RuntimeError(f"Missing {tif_name} or {csv_name} in {bid}.zip")
            tif_meta = _extract_entry_to_file(
                opener, url, bid, total_len, etag, by_base[tif_name], base_dest / "DSLR" / "img" / tif_name, ledger
            )
            csv_meta = _extract_entry_to_file(
                opener, url, bid, total_len, etag, by_base[csv_name], base_dest / "DSLR" / "annotation" / csv_name, ledger
            )
            tif_p = pathlib.Path(tif_meta["path"])
            csv_p = pathlib.Path(csv_meta["path"])
            px_sha, (img_w, img_h) = _decoded_rgb_sha256(tif_p)
            if px_sha in existing_pixel_shas:
                raise RuntimeError(f"Pixel SHA duplicate between {side_stem} and {existing_pixel_shas[px_sha]}")
            existing_pixel_shas[px_sha] = side_stem

            gts, cls_counts, dropped = _validate_csv(csv_p, img_w, img_h)
            rec: Dict[str, Any] = {
                "board_side_id": side_stem,
                "physical_board": bid,
                "role": role,
                "physical_family": spec["physical_family"],
                "family_proof": spec["family_proof"],
                "official_zip_url": url,
                "official_etag": etag,
                "tif_path": str(tif_p),
                "tif_sha256": tif_meta["sha256"],
                "tif_crc32_hex": tif_meta["crc32_hex"],
                "tif_bytes": tif_meta["bytes"],
                "csv_path": str(csv_p),
                "csv_sha256": csv_meta["sha256"],
                "csv_crc32_hex": csv_meta["crc32_hex"],
                "pixel_sha256": px_sha,
                "image_wh": [img_w, img_h],
                "fics6_gt_total": len(gts),
                "class_counts": cls_counts,
                "dropped_non_6class": dropped,
            }
            if role.startswith("codex_blind"):
                side_lbl = side_stem.split("_", 1)[1]
                coco_meta = _write_coco(
                    stem=side_stem,
                    board_id=bid,
                    side=side_lbl,
                    tif_path=tif_p,
                    tif_sha=tif_meta["sha256"],
                    csv_path=csv_p,
                    csv_sha=csv_meta["sha256"],
                    gts=gts,
                    img_w=img_w,
                    img_h=img_h,
                    out_path=BLIND_DIR / f"{side_stem}_official_coco.json",
                )
                rec["coco_path"] = coco_meta["path"]
                rec["coco_sha256"] = coco_meta["sha256"]
                blind_records.append(rec)
            else:
                recovered_sides.append(rec)
            print(f"[OK] {side_stem} ({role}): {img_w}x{img_h}, GT={len(gts)}, counts={cls_counts}, dropped={dropped}", flush=True)

    ledger.save()

    train_inc = {c: sum(int(r["class_counts"][c]) for r in recovered_sides) for c in FICS_SIX_CLASSES}
    blind_manifest = {
        "schema_version": "r17_codex_blind_manifest_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "antigravity_gpu_calls": 0,
        "antigravity_image_previews": 0,
        "boards": blind_records,
    }
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    (BLIND_DIR / "r17_codex_blind_manifest.json").write_text(
        json.dumps(blind_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    audit_doc = {
        "schema_version": "r17_data_expansion_audit_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "network_ledger_path": str(LEDGER_PATH),
        "cumulative_http_bytes_read": ledger.total_bytes,
        "train_expansion_sides_count": len(recovered_sides),
        "train_expansion_physical_boards": sorted({r["physical_board"] for r in recovered_sides}),
        "train_expansion_gt_total": sum(int(r["fics6_gt_total"]) for r in recovered_sides),
        "train_expansion_class_increments": train_inc,
        "train_expansion_sides": recovered_sides,
        "codex_blind_boards": [
            {
                "board_side_id": r["board_side_id"],
                "physical_board": r["physical_board"],
                "physical_family": r["physical_family"],
                "fics6_gt_total": r["fics6_gt_total"],
                "class_counts": r["class_counts"],
                "coco_path": r["coco_path"],
                "coco_sha256": r["coco_sha256"],
                "tif_sha256": r["tif_sha256"],
            }
            for r in blind_records
        ],
        "excluded_boards_with_family_proof": {
            "s5_s6_s7": "Cisco WIC-1DSU-T1 (Altera EPM7128 + IDT/PI 74FCT + DS21721) -> shares Cisco WIC family with val_split (s2, s3) and dev (s1).",
            "s11": "Supermicro (ASPEED AST2400 + EQX68;01ZDE + 2R2;1804) -> shares vendor/silkscreen family with dev (s4: EQX58;01ZDE + 2R2;1804).",
            "s17": "TI MSP430 LaunchPad -> shares MSP430 family with dev (s20).",
            "s10_s16_s19_s31_s9": "Sealed/reserved blind boards & WACV Zybo overlap.",
        },
        "fpic_local_static_audit": _audit_local_fpic(),
        "dedup_check": {"zero_decoded_pixel_sha256_duplicates": True},
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "r17_data_expansion_audit.json").write_text(
        json.dumps(audit_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"DONE R17 fetch & audit. Train increment={train_inc}, total_http_MB={ledger.total_bytes/1e6:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
