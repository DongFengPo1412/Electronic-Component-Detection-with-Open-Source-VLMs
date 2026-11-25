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


class TestRound30GtScoringUi(unittest.TestCase):
    def test_r28_app_js_unwraps_post_hoc_eval_and_guards_finite_values(self) -> None:
        js_path = _SW_ROOT / "scripts" / "static_r28" / "r28_app.js"
        text = js_path.read_text(encoding="utf-8")
        self.assertIn("function unwrapPostHocEval(", text)
        self.assertIn("raw.post_hoc_eval", text)
        self.assertIn("Number.isFinite(", text)
        self.assertIn("function extractPersistedGtScoresFromCmp(", text)
        self.assertIn("cmpLoadSeq", text)
        self.assertIn("cmpGtEvalSeq", text)
        self.assertIn("gt-per-class-drawer", text)
        self.assertIn("formatIdentityBasis(", text)
        # Ensure we do not directly read gtEval.metrics without unwrapping post_hoc_eval
        self.assertNotIn("const m = gtEval.metrics || {};", text)

    def test_export_comparison_includes_post_hoc_eval_when_scored(self) -> None:
        r29_ws = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration30-20261126\live_ws")
        if not r29_ws.is_dir():
            self.skipTest("R30 live_ws not present")
        with tempfile.TemporaryDirectory() as td:
            # Copy only the unscored comparison or test via LiveDemoService on a temp workspace
            svc = LiveDemoService(workspace_root=r29_ws)
            blob_unscored, ctype, _ = svc.export_comparison(
                comparison_id="cmp_20261126_030920_5643df53",
                kind="json",
            )
            self.assertEqual(ctype, "application/json; charset=utf-8")
            doc = json.loads(blob_unscored.decode("utf-8"))
            self.assertEqual(len(doc["slots"]), 3)
            for slot in doc["slots"]:
                self.assertIn("post_hoc_eval", slot)
                self.assertIsNone(slot["post_hoc_eval"])

    def test_evaluate_job_with_coco_rejects_mismatched_dimensions_and_filename(self) -> None:
        r30_ws = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration30-20261126\live_ws")
        if not r30_ws.is_dir():
            self.skipTest("R30 live_ws not present")
        svc = LiveDemoService(workspace_root=r30_ws)
        bad_dim_coco = {
            "images": [{"id": 1, "file_name": "s20_front_c1_2560_3072.png", "width": 512, "height": 512}],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [],
        }
        with self.assertRaises(DemoError) as ctx1:
            svc.evaluate_job_with_coco(job_id="job_20261126_035252_c7444eab", coco_gt=bad_dim_coco)
        self.assertIn("尺寸", str(ctx1.exception))

        bad_name_coco = {
            "images": [{"id": 1, "file_name": "unrelated_board.png", "width": 768, "height": 768}],
            "categories": bad_dim_coco["categories"],
            "annotations": [],
        }
        with self.assertRaises(DemoError) as ctx2:
            svc.evaluate_job_with_coco(job_id="job_20261126_035252_c7444eab", coco_gt=bad_name_coco)
        self.assertIn("缺少身份对应依据", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
