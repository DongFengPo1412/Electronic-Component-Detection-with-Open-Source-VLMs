"""Unit and regression tests for hardened in-repo FICS CPU replay & v2 termination semantics."""

import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from pcb_vlm.config import get_canonical_categories  # noqa: E402
from pcb_vlm.data.fics_dslr import FICS_CANONICAL_CATEGORIES, file_sha256  # noqa: E402
from pcb_vlm.evaluation.fics_replay import (  # noqa: E402
    FICS_SIX_CLASSES,
    classify_call_termination_and_parse_v2,
    evaluate_fics_replay,
    load_fics_six_class_config,
    merge_board_coco_gts_unique_ids,
    validate_call_record_integrity,
    validate_coco_unique_image_ids,
    validate_strict_json_contract,
)
import replay_fics_eval  # noqa: E402

FIXED_PROMPT = (
    "Locate all visible electronic components belonging to these classes: "
    "capacitor, resistor, inductor, transistor, diode, ic. Return a JSON array of objects "
    "with keys bbox_2d and label. Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized "
    "to [0, 1000] relative to this input image. Use exactly one of the listed class names "
    "for each object. If no target components are visible, return []. Do not include explanations."
)
FIXED_PROMPT_SHA = hashlib.sha256(FIXED_PROMPT.encode("utf-8")).hexdigest()


def _make_record(
    *,
    call_id: str,
    model_key: str = "qwen3",
    model_repo_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    model_revision: str | None = "rev-test",
    board_id: str = "s18_front",
    image_id: int = 18,
    mode: str = "W",
    tile_id: int | None = None,
    width: int = 500,
    height: int = 500,
    roi_offset_x: int = 0,
    roi_offset_y: int = 0,
    tile_offset_x: int = 0,
    tile_offset_y: int = 0,
    sha256: str = "a" * 64,
    image_path: str | None = None,
    raw_output: str = "[]",
    ok: bool = True,
    generated_token_ids: list[int] | None = None,
    max_new_tokens: int = 512,
    finish_reason: str | None = "eos_token",
) -> dict:
    toks = [1, 151645] if generated_token_ids is None else list(generated_token_ids)
    rec = {
        "call_id": call_id,
        "board_id": board_id,
        "image_id": image_id,
        "mode": mode,
        "tile_id": tile_id,
        "model_key": model_key,
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "ok": ok,
        "orig_input_width": width,
        "orig_input_height": height,
        "roi_offset_x": roi_offset_x,
        "roi_offset_y": roi_offset_y,
        "tile_offset_x": tile_offset_x,
        "tile_offset_y": tile_offset_y,
        "expected_sha256": sha256,
        "input_sha256": sha256,
        "image_path": image_path,
        "max_new_tokens": max_new_tokens,
        "prompt_sha256": FIXED_PROMPT_SHA,
        "coordinate_protocol": "xyxy_norm1000",
        "raw_output": raw_output,
    }
    if ok:
        rec["generated_token_ids"] = toks
        rec["generated_tokens"] = len(toks)
        rec["finish_reason"] = finish_reason
    return rec


def _make_single_model_plan(
    *,
    model_key: str = "qwen3",
    repo_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    revision: str | None = "rev-test",
    boards: dict | None = None,
    queue_entries: list[dict] | None = None,
) -> dict:
    return {
        "model_execution_order": [model_key],
        "models": {
            model_key: {
                "repo_id": repo_id,
                "revision": revision,
                "visual_budget": {"shortest_edge": 65536, "longest_edge": 200704},
            }
        },
        "inference_config": {
            "prompt": FIXED_PROMPT,
            "prompt_sha256": FIXED_PROMPT_SHA,
            "coordinate_protocol": "xyxy_norm1000",
            "max_new_tokens": 512,
        },
        "boards": boards
        or {
            "s18_front": {"image_id": 18, "orig_wh": [1000, 1000], "gt_count": 1},
        },
        "queues": {
            model_key: queue_entries or [],
        },
    }


class TestFicsReplayV2(unittest.TestCase):
    def test_fics_config_isolated_from_wacv_11_classes(self):
        cfg = load_fics_six_class_config()
        fics_names = [c["name"] for c in cfg["classes"]]
        wacv_names = get_canonical_categories()
        self.assertEqual(fics_names, list(FICS_SIX_CLASSES))
        self.assertEqual(len(fics_names), 6)
        self.assertEqual(len(wacv_names), 11)
        self.assertTrue(set(fics_names).issubset(set(wacv_names)))

    def test_normal_eos_syntax_incomplete_not_counted_as_length_truncated(self):
        out = classify_call_termination_and_parse_v2(
            raw_text='[{"bbox_2d": [100, 100, 200, 200], "label": "capacitor"}, {"bbox_2d": [300, 300',
            width=1000,
            height=1000,
            generated_tokens=3,
            max_new_tokens=512,
            generated_token_ids=[10, 20, 151645],
            eos_token_ids=[151645, 151643],
        )
        self.assertTrue(out["normal_eos"])
        self.assertFalse(out["length_limited_without_eos"])
        self.assertFalse(out["is_truncated"])
        self.assertTrue(out["syntax_or_format_incomplete_after_eos"])
        self.assertTrue(out["legacy_is_truncated_v1"])

    def test_hit_max_new_tokens_without_eos_counted_as_length_truncated(self):
        out = classify_call_termination_and_parse_v2(
            raw_text='[{"bbox_2d": [100, 100, 200, 200], "label": "resistor"}, {"bbox_2d": [300, 300',
            width=1000,
            height=1000,
            generated_tokens=512,
            max_new_tokens=512,
            generated_token_ids=list(range(512)),  # No trailing EOS
            eos_token_ids=[151645, 151643],
        )
        self.assertFalse(out["normal_eos"])
        self.assertTrue(out["length_limited_without_eos"])
        self.assertTrue(out["is_truncated"])
        self.assertFalse(out["syntax_or_format_incomplete_after_eos"])
        self.assertEqual(len(out["objects"]), 1)

    def test_valid_json_closing_at_max_new_tokens_without_eos_retains_valid_schema(self):
        out = classify_call_termination_and_parse_v2(
            raw_text='[{"bbox_2d": [100, 100, 200, 200], "label": "resistor"}]',
            width=1000,
            height=1000,
            generated_tokens=512,
            max_new_tokens=512,
            generated_token_ids=list(range(512)),  # Ends without EOS exactly at budget
            eos_token_ids=[151645, 151643],
        )
        self.assertFalse(out["normal_eos"])
        self.assertTrue(out["length_limited_without_eos"])
        self.assertTrue(out["complete_json_syntax"])
        self.assertTrue(out["strict_schema_ok"])
        self.assertFalse(out["syntax_or_format_incomplete_after_eos"])
        self.assertEqual(len(out["objects"]), 1)

    def test_compatible_extraction_retains_objects_without_strict_schema_ok(self):
        out = classify_call_termination_and_parse_v2(
            raw_text="[{'bbox_2d': [100, 200, 300, 400], 'label': 'diode'}]",
            width=1000,
            height=1000,
            generated_tokens=3,
            max_new_tokens=512,
            generated_token_ids=[5, 6, 151645],
            eos_token_ids=[151645, 151643],
        )
        self.assertEqual(len(out["objects"]), 1)
        self.assertEqual(out["objects"][0]["label"], "diode")
        self.assertFalse(out["strict_schema_ok"])
        self.assertTrue(out["partial_recovery"])
        self.assertEqual(out["call_bucket"], "compatible_non_json_or_partial_with_objects")

    def test_strict_schema_rejects_strings_bool_nan_inf_inverted_and_explicit_empty(self):
        bad_payloads = [
            '[{"bbox_2d": ["100", 100, 200, 200], "label": "capacitor"}]',
            '[{"bbox_2d": [true, 100, 200, 200], "label": "capacitor"}]',
            '[{"bbox_2d": [NaN, 100, 200, 200], "label": "capacitor"}]',
            '[{"bbox_2d": [100, 100, Infinity, 200], "label": "capacitor"}]',
            '[{"bbox_2d": [300, 100, 200, 200], "label": "capacitor"}]',
        ]
        for payload in bad_payloads:
            st = validate_strict_json_contract(payload)
            self.assertFalse(st["strict_schema_ok"], f"Expected strict_schema_ok=False for {payload}")
            v2 = classify_call_termination_and_parse_v2(
                raw_text=payload,
                width=1000,
                height=1000,
                generated_tokens=2,
                max_new_tokens=512,
                generated_token_ids=[1, 151645],
                eos_token_ids=[151645, 151643],
            )
            self.assertFalse(v2["strict_schema_ok"])
            self.assertFalse(v2["explicit_empty"], f"Non-empty bad object must not set explicit_empty: {payload}")

        empty_v2 = classify_call_termination_and_parse_v2(
            raw_text="[]",
            width=1000,
            height=1000,
            generated_tokens=2,
            max_new_tokens=512,
            generated_token_ids=[1, 151645],
            eos_token_ids=[151645, 151643],
        )
        self.assertTrue(empty_v2["strict_schema_ok"])
        self.assertTrue(empty_v2["explicit_empty"])
        self.assertEqual(empty_v2["call_bucket"], "explicit_no_objects")

    def test_intra_board_nms_and_forbid_cross_board_nms(self):
        cats = list(FICS_CANONICAL_CATEGORIES)
        b1_coco = {
            "images": [{"id": 1, "board_id": "s18_front", "width": 1000, "height": 1000}],
            "categories": cats,
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "bbox_xyxy": [100, 100, 200, 200]}],
        }
        b2_coco = {
            "images": [{"id": 1, "board_id": "s28_front", "width": 1000, "height": 1000}],
            "categories": cats,
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "bbox_xyxy": [100, 100, 200, 200]}],
        }
        combined = merge_board_coco_gts_unique_ids([("s18_front", b1_coco, 18), ("s28_front", b2_coco, 28)])

        sha1 = "1" * 64
        sha2 = "2" * 64
        sha3 = "3" * 64
        plan = _make_single_model_plan(
            boards={
                "s18_front": {"image_id": 18, "orig_wh": [1000, 1000], "gt_count": 1},
                "s28_front": {"image_id": 28, "orig_wh": [1000, 1000], "gt_count": 1},
            },
            queue_entries=[
                {"call_id": "m_s18_front_T00", "board_id": "s18_front", "image_id": 18, "mode": "T", "tile_id": 0, "width": 500, "height": 500, "roi_offset_x": 0, "roi_offset_y": 0, "tile_offset_x": 0, "tile_offset_y": 0, "sha256": sha1},
                {"call_id": "m_s18_front_T01", "board_id": "s18_front", "image_id": 18, "mode": "T", "tile_id": 1, "width": 500, "height": 500, "roi_offset_x": 0, "roi_offset_y": 0, "tile_offset_x": 50, "tile_offset_y": 50, "sha256": sha2},
                {"call_id": "m_s28_front_T00", "board_id": "s28_front", "image_id": 28, "mode": "T", "tile_id": 0, "width": 500, "height": 500, "roi_offset_x": 0, "roi_offset_y": 0, "tile_offset_x": 0, "tile_offset_y": 0, "sha256": sha3},
            ],
        )
        rows = [
            _make_record(call_id="m_s18_front_T00", board_id="s18_front", image_id=18, mode="T", tile_id=0, sha256=sha1, raw_output='[{"bbox_2d": [200, 200, 400, 400], "label": "capacitor"}]'),
            _make_record(call_id="m_s18_front_T01", board_id="s18_front", image_id=18, mode="T", tile_id=1, tile_offset_x=50, tile_offset_y=50, sha256=sha2, raw_output='[{"bbox_2d": [100, 100, 300, 300], "label": "capacitor"}]'),
            _make_record(call_id="m_s28_front_T00", board_id="s28_front", image_id=28, mode="T", tile_id=0, sha256=sha3, raw_output='[{"bbox_2d": [200, 200, 400, 400], "label": "capacitor"}]'),
        ]
        gt_sha = hashlib.sha256(json.dumps(combined, sort_keys=True).encode("utf-8")).hexdigest()
        res = evaluate_fics_replay(
            combined_coco=combined,
            call_rows_by_model={"qwen3": rows},
            frozen_plan=plan,
            expected_gt_sha256=gt_sha,
            actual_gt_sha256=gt_sha,
            run_coco_ap=False,
        )
        m = res["models"]["qwen3"]
        self.assertEqual(m["per_board"]["s18_front"]["T"]["raw_global_predictions"], 2)
        self.assertEqual(m["per_board"]["s18_front"]["T"]["post_nms_predictions"], 1)
        self.assertEqual(m["per_board"]["s28_front"]["T"]["post_nms_predictions"], 1)
        self.assertEqual(m["combined"]["T"]["evaluate_detections_iou50"]["tp"], 2)
        self.assertEqual(m["combined"]["T"]["evaluate_detections_iou50"]["fp"], 0)
        self.assertEqual(m["combined"]["T"]["evaluate_detections_iou50"]["fn"], 0)

    def test_combined_coco_rejects_colliding_image_ids_and_incompatible_categories(self):
        colliding = {
            "images": [
                {"id": 1, "board_id": "s18_front", "width": 800, "height": 600},
                {"id": 1, "board_id": "s28_front", "width": 800, "height": 600},
            ],
            "categories": list(FICS_CANONICAL_CATEGORIES),
            "annotations": [],
        }
        with self.assertRaises(ValueError):
            validate_coco_unique_image_ids(colliding)

        a = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "capacitor"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}],
        }
        b = copy.deepcopy(a)
        b["categories"][0]["name"] = "resistor"
        with self.assertRaises(ValueError):
            merge_board_coco_gts_unique_ids([("a", a, 18), ("b", b, 28)])

        c = copy.deepcopy(a)
        c["annotations"][0]["category_id"] = 99
        with self.assertRaises(ValueError):
            merge_board_coco_gts_unique_ids([("a", a, 18), ("c", c, 28)])

    def test_incomplete_or_failed_prediction_never_scored_as_zero(self):
        b1_coco = {
            "images": [{"id": 18, "board_id": "s18_front", "width": 1000, "height": 1000}],
            "categories": list(FICS_CANONICAL_CATEGORIES),
            "annotations": [{"id": 1, "image_id": 18, "category_id": 1, "bbox": [100, 100, 100, 100], "bbox_xyxy": [100, 100, 200, 200]}],
        }
        sha_w = "a" * 64
        plan = _make_single_model_plan(
            model_key="qwen2",
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            revision=None,
            queue_entries=[
                {"call_id": "qwen2_s18_front_W", "board_id": "s18_front", "image_id": 18, "mode": "W", "tile_id": None, "width": 1000, "height": 1000, "roi_offset_x": 0, "roi_offset_y": 0, "tile_offset_x": 0, "tile_offset_y": 0, "sha256": sha_w}
            ],
        )
        gt_sha = "f" * 64
        failed_row = _make_record(
            call_id="qwen2_s18_front_W",
            model_key="qwen2",
            model_repo_id="Qwen/Qwen2-VL-2B-Instruct",
            model_revision=None,
            width=1000,
            height=1000,
            sha256=sha_w,
            ok=False,
        )
        failed_row["is_timeout"] = True
        res_fail = evaluate_fics_replay(
            combined_coco=b1_coco,
            call_rows_by_model={"qwen2": [failed_row]},
            frozen_plan=plan,
            expected_gt_sha256=gt_sha,
            actual_gt_sha256=gt_sha,
            run_coco_ap=False,
        )
        self.assertFalse(res_fail["models"]["qwen2"]["model_technical_complete"])
        self.assertFalse(res_fail["completeness_verified"])
        self.assertEqual(res_fail["models"]["qwen2"]["combined"]["W"]["evaluation_status"], "incomplete_technical_failure")
        self.assertNotIn("evaluate_detections_iou50", res_fail["models"]["qwen2"]["combined"]["W"])

        res_miss = evaluate_fics_replay(
            combined_coco=b1_coco,
            call_rows_by_model={"qwen2": []},
            frozen_plan=plan,
            expected_gt_sha256=gt_sha,
            actual_gt_sha256=gt_sha,
            run_coco_ap=False,
        )
        self.assertFalse(res_miss["models"]["qwen2"]["model_technical_complete"])
        self.assertEqual(res_miss["models"]["qwen2"]["per_board"]["s18_front"]["W"]["evaluation_status"], "incomplete_technical_failure")

    def test_rejects_prompt_sha_input_hash_and_missing_required_fields(self):
        with self.assertRaises(ValueError):
            validate_call_record_integrity(
                _make_record(call_id="c1") | {"prompt_sha256": "0" * 64},
                expected_prompt_sha256=FIXED_PROMPT_SHA,
            )

        with self.assertRaises(ValueError):
            validate_call_record_integrity(
                _make_record(call_id="c2") | {"expected_sha256": "a" * 64, "input_sha256": "b" * 64},
                expected_prompt_sha256=FIXED_PROMPT_SHA,
            )

        # Missing required identity fields must be rejected even with verify_disk_hashes=True
        missing_row = _make_record(call_id="c_miss")
        for k in ("input_sha256", "expected_sha256", "coordinate_protocol"):
            missing_row.pop(k, None)
        with self.assertRaises(ValueError):
            validate_call_record_integrity(missing_row, expected_prompt_sha256=FIXED_PROMPT_SHA, verify_disk_hashes=True)

        with tempfile.TemporaryDirectory() as tmp:
            img_p = Path(tmp) / "tile.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(img_p)
            real_sha = file_sha256(img_p)
            ok_disk_row = _make_record(call_id="c3", width=32, height=32, sha256=real_sha, image_path=str(img_p))
            self.assertTrue(
                validate_call_record_integrity(ok_disk_row, expected_prompt_sha256=FIXED_PROMPT_SHA, verify_disk_hashes=True)
            )
            # Wrong on-disk dimensions rejected
            bad_dim_row = _make_record(call_id="c3_dim", width=64, height=32, sha256=real_sha, image_path=str(img_p))
            with self.assertRaises(ValueError):
                validate_call_record_integrity(bad_dim_row, expected_prompt_sha256=FIXED_PROMPT_SHA, verify_disk_hashes=True)
            # Token-length / finish_reason mismatch rejected
            bad_tok_len = copy.deepcopy(ok_disk_row)
            bad_tok_len["generated_tokens"] = 99
            with self.assertRaises(ValueError):
                validate_call_record_integrity(bad_tok_len, expected_prompt_sha256=FIXED_PROMPT_SHA)
            bad_finish = copy.deepcopy(ok_disk_row)
            bad_finish["generated_token_ids"] = [1, 2]
            bad_finish["generated_tokens"] = 2
            bad_finish["finish_reason"] = "eos_token"
            with self.assertRaises(ValueError):
                validate_call_record_integrity(bad_finish, expected_prompt_sha256=FIXED_PROMPT_SHA, expected_eos_token_ids=[151645, 151643])
            # Corrupt disk bytes rejected
            Image.new("RGB", (32, 32), (200, 10, 10)).save(img_p)
            with self.assertRaises(ValueError):
                validate_call_record_integrity(ok_disk_row, expected_prompt_sha256=FIXED_PROMPT_SHA, verify_disk_hashes=True)

    def test_rejects_tampered_config_prompt_or_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg_base = load_fics_six_class_config()
            bad_prompt_cfg = copy.deepcopy(cfg_base)
            bad_prompt_cfg["protocol"]["prompt"] += " extra"
            bp_path = tdir / "bad_prompt.yaml"
            bp_path.write_text(json.dumps(bad_prompt_cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fics_six_class_config(bp_path)

            bad_cls_cfg = copy.deepcopy(cfg_base)
            bad_cls_cfg["classes"][0]["name"] = "resistor"
            bc_path = tdir / "bad_cls.yaml"
            bc_path.write_text(json.dumps(bad_cls_cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fics_six_class_config(bc_path)

    def test_rejects_frozen_plan_mismatches_duplicate_calls_and_self_matching_nonfrozen_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            img_p = tdir / "board_w.png"
            Image.new("RGB", (1000, 1000), (20, 40, 60)).save(img_p)
            real_sha = file_sha256(img_p)
            coco = {
                "images": [{"id": 18, "board_id": "s18_front", "width": 1000, "height": 1000}],
                "categories": list(FICS_CANONICAL_CATEGORIES),
                "annotations": [{"id": 1, "image_id": 18, "category_id": 1, "bbox": [100, 100, 100, 100], "bbox_xyxy": [100, 100, 200, 200]}],
            }
            gt_sha = "e" * 64
            q_entry = {
                "call_id": "qwen3_s18_front_W",
                "board_id": "s18_front",
                "image_id": 18,
                "mode": "W",
                "tile_id": None,
                "width": 1000,
                "height": 1000,
                "roi_offset_x": 0,
                "roi_offset_y": 0,
                "tile_offset_x": 0,
                "tile_offset_y": 0,
                "sha256": real_sha,
                "path": str(img_p),
            }
            plan = _make_single_model_plan(queue_entries=[q_entry])
            good_row = _make_record(
                call_id="qwen3_s18_front_W",
                width=1000,
                height=1000,
                sha256=real_sha,
                image_path=str(img_p),
                raw_output='[{"bbox_2d": [100, 100, 200, 200], "label": "capacitor"}]',
            )

            # Baseline valid passes
            res = evaluate_fics_replay(
                combined_coco=coco,
                call_rows_by_model={"qwen3": [good_row]},
                frozen_plan=plan,
                expected_gt_sha256=gt_sha,
                actual_gt_sha256=gt_sha,
                verify_disk_hashes=True,
                run_coco_ap=False,
            )
            self.assertTrue(res["completeness_verified"])

            # Duplicate call_id
            with self.assertRaises(ValueError):
                evaluate_fics_replay(
                    combined_coco=coco,
                    call_rows_by_model={"qwen3": [good_row, copy.deepcopy(good_row)]},
                    frozen_plan=plan,
                    expected_gt_sha256=gt_sha,
                    actual_gt_sha256=gt_sha,
                    run_coco_ap=False,
                )

            # Unexpected extra call_id
            extra = _make_record(call_id="qwen3_unexpected_T99", width=1000, height=1000, sha256=real_sha)
            with self.assertRaises(ValueError):
                evaluate_fics_replay(
                    combined_coco=coco,
                    call_rows_by_model={"qwen3": [good_row, extra]},
                    frozen_plan=plan,
                    expected_gt_sha256=gt_sha,
                    actual_gt_sha256=gt_sha,
                    run_coco_ap=False,
                )

            # Tampered fields caught by frozen plan binding (even if two row SHAs match each other!)
            for field, bad_val in [
                ("tile_offset_x", 5),
                ("roi_offset_y", 7),
                ("orig_input_width", 999),
                ("model_key", "deepseek"),
                ("model_repo_id", "Qwen/Wrong-Model"),
                ("max_new_tokens", 128),
                ("coordinate_protocol", "yxyx_norm1000"),
                ("input_sha256", "c" * 64),
            ]:
                bad = copy.deepcopy(good_row)
                bad[field] = bad_val
                if field == "input_sha256":
                    # Both self-declared SHAs equal each other but differ from frozen queue sha256
                    bad["expected_sha256"] = bad_val
                with self.assertRaises(ValueError, msg=f"Expected tamper rejection for {field}"):
                    evaluate_fics_replay(
                        combined_coco=coco,
                        call_rows_by_model={"qwen3": [bad]},
                        frozen_plan=plan,
                        expected_gt_sha256=gt_sha,
                        actual_gt_sha256=gt_sha,
                        run_coco_ap=False,
                    )

    def test_cli_anti_overwrite_incomplete_diagnostic_and_portable_remapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            img_dir = tdir / "relocated_imgs"
            img_dir.mkdir()
            img_p = img_dir / "s18_w.png"
            Image.new("RGB", (1000, 1000), (15, 25, 35)).save(img_p)
            real_sha = file_sha256(img_p)

            coco = {
                "images": [{"id": 18, "board_id": "s18_front", "width": 1000, "height": 1000}],
                "categories": list(FICS_CANONICAL_CATEGORIES),
                "annotations": [{"id": 1, "image_id": 18, "category_id": 1, "bbox": [100, 100, 100, 100], "bbox_xyxy": [100, 100, 200, 200]}],
            }
            coco_p = tdir / "coco.json"
            coco_p.write_text(json.dumps(coco), encoding="utf-8")
            gt_sha = file_sha256(coco_p)

            plan = {
                "model_execution_order": ["qwen3", "qwen2"],
                "models": {
                    "qwen3": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": "rev-test"},
                    "qwen2": {"repo_id": "Qwen/Qwen2-VL-2B-Instruct", "revision": None},
                },
                "inference_config": {
                    "prompt": FIXED_PROMPT,
                    "prompt_sha256": FIXED_PROMPT_SHA,
                    "coordinate_protocol": "xyxy_norm1000",
                    "max_new_tokens": 512,
                },
                "boards": {"s18_front": {"image_id": 18, "orig_wh": [1000, 1000], "gt_count": 1}},
                "queues": {
                    "formal_67": [
                        {
                            "board_id": "s18_front",
                            "tag": "W",
                            "mode": "W",
                            "image_id": 18,
                            "tile_id": None,
                            "width": 1000,
                            "height": 1000,
                            "roi_offset_x": 0,
                            "roi_offset_y": 0,
                            "tile_offset_x": 0,
                            "tile_offset_y": 0,
                            "sha256": real_sha,
                            "path": r"C:\NonExistent\OldMachine\cache\s18_w.png",
                        }
                    ]
                },
            }
            plan_p = tdir / "frozen_plan.json"
            plan_p.write_text(json.dumps(plan), encoding="utf-8")

            runs = tdir / "runs"
            (runs / "run_qwen3").mkdir(parents=True)
            (runs / "run_qwen2").mkdir(parents=True)
            q3_row = _make_record(
                call_id="qwen3_formal_s18_front_W",
                model_key="qwen3",
                model_repo_id="Qwen/Qwen3-VL-2B-Instruct",
                model_revision="rev-test",
                width=1000,
                height=1000,
                sha256=real_sha,
                image_path=r"C:\NonExistent\OldMachine\cache\s18_w.png",
                raw_output='[{"bbox_2d": [100, 100, 200, 200], "label": "capacitor"}]',
            )
            q2_row = _make_record(
                call_id="qwen2_formal_s18_front_W",
                model_key="qwen2",
                model_repo_id="Qwen/Qwen2-VL-2B-Instruct",
                model_revision=None,
                width=1000,
                height=1000,
                sha256=real_sha,
                image_path=r"C:\NonExistent\OldMachine\cache\s18_w.png",
                raw_output="[]",
            )
            q3_raw = runs / "run_qwen3" / "raw_outputs_all.jsonl"
            q2_raw = runs / "run_qwen2" / "raw_outputs_all.jsonl"
            q3_raw.write_text(json.dumps(q3_row) + "\n", encoding="utf-8")
            q2_raw.write_text(json.dumps(q2_row) + "\n", encoding="utf-8")

            raw_sha_before = file_sha256(q3_raw)
            coco_sha_before = file_sha256(coco_p)
            plan_sha_before = file_sha256(plan_p)

            # 1. Anti-overwrite: --output == raw, GT, plan, config, case-variant, hardlink, pre-existing output
            cfg_copy = tdir / "fics_cfg.yaml"
            cfg_copy.write_text((_REPO_ROOT / "configs" / "fics_six_classes.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            existing_out = tdir / "existing.json"
            existing_out.write_text('{"keep": true}\n', encoding="utf-8")
            existing_sha_before = file_sha256(existing_out)
            case_raw_out = q3_raw.parent / q3_raw.name.upper()
            hardlink_out = tdir / "hardlink_to_raw.jsonl"
            collide_targets = [q3_raw, coco_p, plan_p, cfg_copy, case_raw_out, existing_out]
            try:
                os.link(q3_raw, hardlink_out)
                collide_targets.append(hardlink_out)
            except OSError:
                pass
            for collide_target in collide_targets:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rc = replay_fics_eval.main([
                        "--gt-coco", str(coco_p),
                        "--runs-dir", str(runs),
                        "--config", str(cfg_copy),
                        "--plan", str(plan_p),
                        "--expected-gt-sha256", gt_sha,
                        "--no-coco-ap",
                        "--output", str(collide_target),
                    ])
                self.assertNotEqual(rc, 0)
            self.assertEqual(file_sha256(q3_raw), raw_sha_before)
            self.assertEqual(file_sha256(coco_p), coco_sha_before)
            self.assertEqual(file_sha256(plan_p), plan_sha_before)
            self.assertEqual(file_sha256(existing_out), existing_sha_before)

            # Duplicate --raw-jsonl key rejected
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                rc_dup = replay_fics_eval.main([
                    "--gt-coco", str(coco_p),
                    "--raw-jsonl", f"qwen3={q3_raw}",
                    "--raw-jsonl", f"qwen3={q2_raw}",
                    "--plan", str(plan_p),
                    "--expected-gt-sha256", gt_sha,
                    "--no-coco-ap",
                    "--output", str(tdir / "dup_out.json"),
                ])
            self.assertNotEqual(rc_dup, 0)
            self.assertFalse((tdir / "dup_out.json").exists())

            # Formal mode without --expected-gt-sha256 or with --models subset without --diagnostic must fail
            for extra_args in ([], ["--expected-gt-sha256", gt_sha, "--models", "qwen3"], ["--expected-gt-sha256", "0" * 64]):
                err_buf = io.StringIO()
                with contextlib.redirect_stderr(err_buf):
                    rc_bad = replay_fics_eval.main([
                        "--gt-coco", str(coco_p),
                        "--runs-dir", str(runs),
                        "--plan", str(plan_p),
                        "--no-coco-ap",
                        "--output", str(tdir / "blocked_out.json"),
                        *extra_args,
                    ])
                self.assertNotEqual(rc_bad, 0)
                self.assertFalse((tdir / "blocked_out.json").exists())

            # Complete formal replay with portable --path-map + --verify-disk-hashes succeeds with exit 0
            # Note qwen2 returned genuine "[]" -> cleanly gets 0 TP and complete=True
            formal_out = tdir / "nested" / "formal_out.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_ok = replay_fics_eval.main([
                    "--gt-coco", str(coco_p),
                    "--runs-dir", str(runs),
                    "--plan", str(plan_p),
                    "--expected-gt-sha256", gt_sha,
                    "--verify-disk-hashes",
                    "--path-map", rf"C:\NonExistent\OldMachine\cache={img_dir}",
                    "--no-coco-ap",
                    "--output", str(formal_out),
                ])
            self.assertEqual(rc_ok, 0)
            payload = json.loads(formal_out.read_text(encoding="utf-8"))
            self.assertTrue(payload["completeness_verified"])
            self.assertTrue(payload["gt_identity_verified"])
            self.assertTrue(payload["disk_images_reverified"])
            self.assertEqual(payload["disk_images_verified_count"], 2)
            self.assertEqual(payload["models"]["qwen3"]["combined"]["W"]["evaluate_detections_iou50"]["tp"], 1)
            self.assertEqual(payload["models"]["qwen2"]["combined"]["W"]["evaluate_detections_iou50"]["tp"], 0)
            self.assertTrue(payload["models"]["qwen2"]["model_technical_complete"])

            # Missing model directory in formal mode -> non-zero exit + diagnostic incomplete report saved
            incomplete_out = tdir / "incomplete_missing_model.json"
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                rc_inc = replay_fics_eval.main([
                    "--gt-coco", str(coco_p),
                    "--raw-jsonl", f"qwen3={q3_raw}",
                    "--plan", str(plan_p),
                    "--expected-gt-sha256", gt_sha,
                    "--no-coco-ap",
                    "--output", str(incomplete_out),
                ])
            self.assertNotEqual(rc_inc, 0)
            self.assertIn("INCOMPLETE", buf2.getvalue())
            inc_data = json.loads(incomplete_out.read_text(encoding="utf-8"))
            self.assertFalse(inc_data["completeness_verified"])
            self.assertEqual(inc_data["scope"]["missing_models"], ["qwen2"])
            self.assertFalse(inc_data["models"]["qwen2"]["model_technical_complete"])

            # Explicit single-model diagnostic -> scoped diagnostic marked NOT full-matrix complete
            diag_out = tdir / "diag_qwen3.json"
            buf3 = io.StringIO()
            with contextlib.redirect_stdout(buf3):
                rc_diag = replay_fics_eval.main([
                    "--gt-coco", str(coco_p),
                    "--raw-jsonl", f"qwen3={q3_raw}",
                    "--plan", str(plan_p),
                    "--expected-gt-sha256", gt_sha,
                    "--diagnostic",
                    "--models", "qwen3",
                    "--no-coco-ap",
                    "--output", str(diag_out),
                ])
            self.assertEqual(rc_diag, 0)
            diag_data = json.loads(diag_out.read_text(encoding="utf-8"))
            self.assertEqual(diag_data["replay_mode"], "diagnostic")
            self.assertFalse(diag_data["completeness_verified"])
            self.assertFalse(diag_data["scope"]["is_full_planned_matrix"])
            self.assertTrue(diag_data["scoped_subset_complete"])

    def test_watchdog_timeout_and_eof_raw_replay_cleanly_while_tamper_rejected(self) -> None:
        if str(_REPO_ROOT / "tests") not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT / "tests"))
        from pcb_vlm.inference.fics_matrix import run_sequential_matrix
        from test_fics_matrix_runner import _make_sample_plan, _write_fake_worker

        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=3)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            # Replace dummy text bytes with real valid 640x480 PNGs and sync hashes so --verify-disk-hashes runs end-to-end
            for mk in ("qwen2", "qwen3"):
                for q in plan["queues"][mk]:
                    qp = Path(q["path"])
                    Image.new("RGB", (int(q["width"]), int(q["height"])), color=(10, 20, 30)).save(qp)
                    q["sha256"] = file_sha256(qp)
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            # Add coco GT matching the plan's board_id and dims so formal replay can evaluate it
            first_q = plan["queues"]["qwen2"][0]
            coco = {
                "images": [{"id": first_q["image_id"], "file_name": "test_board.png", "width": 100, "height": 100, "board_id": first_q["board_id"]}],
                "categories": [
                    {"id": 1, "name": "capacitor"},
                    {"id": 2, "name": "resistor"},
                    {"id": 3, "name": "inductor"},
                    {"id": 4, "name": "transistor"},
                    {"id": 5, "name": "diode"},
                    {"id": 6, "name": "ic"},
                ],
                "annotations": [{"id": 1, "image_id": first_q["image_id"], "category_id": 6, "bbox": [10, 10, 20, 20]}],
            }
            coco_p = tpath / "gt_coco.json"
            coco_p.write_text(json.dumps(coco), encoding="utf-8")
            gt_sha = hashlib.sha256(coco_p.read_bytes()).hexdigest()

            worker = _write_fake_worker(tpath)
            for scenario, mode_arg in (
                ("pre_input_timeout", "hang_second"),
                ("verified_timeout", "hang_after_verified"),
                ("pre_input_eof", "crash_second"),
            ):
                res = run_sequential_matrix(
                    plan_path=plan_path,
                    output_dir=tpath / f"run_{scenario}",
                    models_filter=["qwen2"],
                    dry_run=False,
                    per_call_timeout_s=0.4,
                    gpu_gate_fn=lambda s, m: {"passed": True},
                    worker_cmd_builder=lambda mk, _mc, _rp, m=mode_arg: [sys.executable, "-u", str(worker), m, mk],
                )
                self.assertEqual(res["exit_code"], 1)
                raw_p = Path(res["stage_summaries"]["qwen2"]["raw_jsonl"])
                out_p = tpath / f"replay_{scenario}.json"
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = replay_fics_eval.main([
                        "--gt-coco", str(coco_p),
                        "--raw-jsonl", f"qwen2={raw_p}",
                        "--plan", str(plan_path),
                        "--expected-gt-sha256", gt_sha,
                        "--verify-disk-hashes",
                        "--no-coco-ap",
                        "--output", str(out_p),
                    ])
                self.assertNotEqual(rc, 0)
                rep = json.loads(out_p.read_text(encoding="utf-8"))
                self.assertFalse(rep["completeness_verified"])
                self.assertFalse(rep["models"]["qwen2"]["model_technical_complete"])
                unit_t = rep["models"]["qwen2"]["per_board"][first_q["board_id"]]["T"]
                self.assertEqual(unit_t["evaluation_status"], "incomplete_technical_failure")
                self.assertFalse(unit_t["technical_complete"])
                self.assertEqual(unit_t["failed_call_ids"], ["custom_call_1"])
                self.assertEqual(unit_t["missing_call_ids"], ["custom_call_2"])
                self.assertEqual(
                    rep["models"]["qwen2"]["combined"]["W"]["evaluation_status"],
                    "incomplete_technical_failure",
                )
                detail = unit_t["failed_call_details"][0]
                if scenario == "verified_timeout":
                    self.assertEqual(detail["input_verification_status"], "verified_by_worker_before_failure")
                else:
                    self.assertEqual(detail["input_verification_status"], "unverified_pre_input_failure")

                # Negative probes against the replay validator using records derived from this raw file
                rows = [json.loads(ln) for ln in raw_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                fail_row = dict(rows[1])
                # 1) ok=True with null input_sha256 must STILL be rejected!
                bad1 = dict(fail_row, ok=True, generated_tokens=0, generated_token_ids=[], finish_reason="eos_token", input_sha256=None)
                with self.assertRaises(ValueError):
                    validate_call_record_integrity(bad1, expected_model_key="qwen2")
                # 2) Arbitrarily missing input_sha256 key on non-watchdog failure must be rejected!
                bad2 = dict(fail_row, source="worker", input_verification_status=None)
                bad2.pop("input_sha256", None)
                with self.assertRaises(ValueError):
                    validate_call_record_integrity(bad2, expected_model_key="qwen2")
                # 3) Present but mismatched input_sha256 must STILL be rejected!
                bad3 = dict(fail_row, input_sha256="0" * 64)
                with self.assertRaises(ValueError):
                    validate_call_record_integrity(bad3, expected_model_key="qwen2")
                # 4) Tampered expected_sha256 vs frozen spec must STILL be rejected!
                bad4 = dict(fail_row, expected_sha256="f" * 64, input_sha256=None)
                spec_map = {q["call_id"]: q for q in plan["queues"]["qwen2"]}
                with self.assertRaises(ValueError):
                    validate_call_record_integrity(bad4, expected_model_key="qwen2", expected_call_spec=spec_map[bad4["call_id"]])


if __name__ == "__main__":
    unittest.main()
