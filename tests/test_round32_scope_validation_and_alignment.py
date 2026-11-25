import copy
import json
import pathlib
import sys
import tempfile
import unittest

_SW_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _SW_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.final_demo_service import DemoError, LiveDemoService  # noqa: E402


def _make_base_coco() -> dict:
    return {
        "eval_scope": "partial_windows",
        "eval_windows": [
            {"window_id": "w1", "bbox_xyxy": [0, 0, 500, 500]},
            {"window_id": "w2", "bbox_xyxy": [500, 0, 1000, 500]},
        ],
        "images": [{"id": 1, "file_name": "board.png", "width": 1000, "height": 1000, "sha256": "b" * 64}],
        "categories": [
            {"id": 1, "name": "capacitor"},
            {"id": 2, "name": "resistor"},
            {"id": 3, "name": "inductor"},
            {"id": 4, "name": "transistor"},
            {"id": 5, "name": "diode"},
            {"id": 6, "name": "ic"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [100.0, 100.0, 40.0, 20.0], "area": 800.0, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [600.0, 100.0, 40.0, 20.0], "area": 800.0, "iscrowd": 0},
        ],
    }


def _setup_service_with_job(td: str) -> tuple[LiveDemoService, str]:
    svc = LiveDemoService(workspace_root=pathlib.Path(td))
    jid = "job_test_r32"
    jdir = svc.jobs_dir / jid
    jdir.mkdir(parents=True, exist_ok=True)
    res_doc = {
        "job_id": jid,
        "upload_id": "up_test",
        "board_id": "board",
        "safe_filename": "board.png",
        "source_image_sha256": "b" * 64,
        "orig_wh": [1000, 1000],
        "mode": "COMPOSITE_YOLO",
        "predictions": [
            {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 140.0, 120.0], "score": 0.9},
            {"label": "capacitor", "bbox_xyxy": [600.0, 100.0, 640.0, 120.0], "score": 0.9},
            # Border-center box at x=500.0: belongs uniquely to w2 [500, 1000), not double-counted
            {"label": "resistor", "bbox_xyxy": [490.0, 100.0, 510.0, 120.0], "score": 0.8},
            # Outside declared windows (y=800): must be excluded in partial_windows
            {"label": "resistor", "bbox_xyxy": [100.0, 800.0, 140.0, 820.0], "score": 0.95},
        ],
        "pre_verifier_candidates": [
            {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 140.0, 120.0], "score": 0.9},
            {"label": "resistor", "bbox_xyxy": [100.0, 800.0, 140.0, 820.0], "score": 0.95},
        ],
        "class_counts": {"resistor": 3, "capacitor": 1},
    }
    (jdir / "job_result.json").write_text(json.dumps(res_doc), encoding="utf-8")
    return svc, jid


class TestRound32ScopeValidationAndAlignment(unittest.TestCase):
    def test_valid_partial_windows_and_pre_verifier_filter_and_rejection_preserves_prior(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc, jid = _setup_service_with_job(td)
            valid = _make_base_coco()
            ev = svc.evaluate_job_with_coco(job_id=jid, coco_gt=valid)
            phe = ev["post_hoc_eval"]
            self.assertEqual(phe["eval_scope"], "partial_windows")
            self.assertFalse(phe["whole_board_f1_valid"])
            self.assertEqual(phe["metrics"]["total_predictions"], 3)
            self.assertEqual(phe["metrics"]["tp"], 2)
            self.assertEqual(phe["metrics"]["fp"], 1)
            self.assertEqual(phe["pre_verifier_metrics"]["total_predictions"], 1)

            # Now test all invalid variants -> HTTP 400 and persisted valid score untouched
            bad_cases = []
            c1 = _make_base_coco(); c1["eval_windows"] = []
            bad_cases.append(("empty_windows", c1))
            c2 = _make_base_coco(); c2["eval_windows"] = [{"window_id": "bad", "bbox_xyxy": [0, 0]}]
            bad_cases.append(("bad_shape", c2))
            c3 = _make_base_coco(); c3["eval_scope"] = "partial_window"
            bad_cases.append(("misspelled_scope", c3))
            c4 = _make_base_coco(); c4["eval_windows"] = {"w1": [0, 0, 500, 500]}
            bad_cases.append(("wrong_container", c4))
            c5 = _make_base_coco(); c5["eval_windows"][0]["bbox_xyxy"] = [0, 0, float("nan"), 500]
            bad_cases.append(("non_finite", c5))
            c6 = _make_base_coco(); c6["eval_windows"][0]["bbox_xyxy"] = [500, 0, 100, 500]
            bad_cases.append(("inverted_box", c6))
            c7 = _make_base_coco(); c7["eval_windows"][0]["bbox_xyxy"] = [0, 0, 1200, 500]
            bad_cases.append(("out_of_bounds", c7))
            c8 = _make_base_coco(); c8["eval_windows"][1]["window_id"] = "w1"
            bad_cases.append(("duplicate_id", c8))
            c9 = _make_base_coco(); c9["eval_windows"][1]["bbox_xyxy"] = [400, 0, 900, 500]
            bad_cases.append(("overlapping_windows", c9))
            c10 = _make_base_coco(); c10["images"][0]["eval_scope"] = "full_image"
            bad_cases.append(("level_conflict", c10))
            c11 = _make_base_coco(); c11["eval_scope"] = "full_image"
            bad_cases.append(("full_image_with_windows", c11))
            c12 = _make_base_coco(); c12["annotations"][0]["bbox"] = [100.0, 800.0, 40.0, 20.0]
            bad_cases.append(("gt_outside_windows", c12))

            for name, bad_coco in bad_cases:
                with self.subTest(case=name):
                    with self.assertRaises(DemoError) as ctx:
                        svc.evaluate_job_with_coco(job_id=jid, coco_gt=bad_coco)
                    self.assertEqual(ctx.exception.status_code, 400)

            # Persisted valid score must remain intact
            persisted = svc.get_job_result(jid)["post_hoc_eval"]
            self.assertEqual(persisted["eval_scope"], "partial_windows")
            self.assertEqual(persisted["metrics"]["tp"], 2)

    def test_standard_full_image_coco_without_scope_fields_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc, jid = _setup_service_with_job(td)
            std_coco = copy.deepcopy(_make_base_coco())
            std_coco.pop("eval_scope", None)
            std_coco.pop("eval_windows", None)
            ev = svc.evaluate_job_with_coco(job_id=jid, coco_gt=std_coco)
            phe = ev["post_hoc_eval"]
            self.assertEqual(phe["eval_scope"], "full_image")
            self.assertTrue(phe["whole_board_f1_valid"])
            self.assertEqual(phe["metrics"]["total_predictions"], 4)


if __name__ == "__main__":
    unittest.main()
