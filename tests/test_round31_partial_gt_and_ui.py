import json
import pathlib
import sys
import tempfile
import unittest

_SW_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _SW_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402


class TestRound31PartialGtAndUi(unittest.TestCase):
    def test_partial_windows_gt_scope_filters_predictions_and_flags_non_whole_board(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            svc = LiveDemoService(workspace_root=ws)
            jid = "job_test_partial"
            jdir = svc.jobs_dir / jid
            jdir.mkdir(parents=True, exist_ok=True)
            res_doc = {
                "job_id": jid,
                "upload_id": "up_test",
                "board_id": "test_board",
                "safe_filename": "test_board.png",
                "source_image_sha256": "a" * 64,
                "orig_wh": [2000, 2000],
                "mode": "COMPOSITE_YOLO",
                "predictions": [
                    {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 150.0, 130.0], "score": 0.9},
                    {"label": "resistor", "bbox_xyxy": [1500.0, 1500.0, 1550.0, 1530.0], "score": 0.9},
                ],
                "class_counts": {"resistor": 2},
            }
            (jdir / "job_result.json").write_text(json.dumps(res_doc), encoding="utf-8")

            coco_gt = {
                "eval_scope": "partial_windows",
                "eval_windows": [{"window_id": "w1_0_0", "bbox_xyxy": [0, 0, 768, 768]}],
                "images": [{"id": 1, "file_name": "test_board.png", "width": 2000, "height": 2000, "sha256": "a" * 64}],
                "categories": [
                    {"id": 1, "name": "capacitor"},
                    {"id": 2, "name": "resistor"},
                    {"id": 3, "name": "inductor"},
                    {"id": 4, "name": "transistor"},
                    {"id": 5, "name": "diode"},
                    {"id": 6, "name": "ic"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 2, "bbox": [100.0, 100.0, 50.0, 30.0], "area": 1500.0, "iscrowd": 0},
                ],
            }
            ev = svc.evaluate_job_with_coco(job_id=jid, coco_gt=coco_gt)
            phe = ev["post_hoc_eval"]
            self.assertEqual(phe["eval_scope"], "partial_windows")
            self.assertFalse(phe["whole_board_f1_valid"])
            self.assertEqual(phe["metrics"]["tp"], 1)
            self.assertEqual(phe["metrics"]["fp"], 0)
            self.assertEqual(phe["metrics"]["fn"], 0)
            self.assertEqual(phe["metrics"]["total_predictions"], 1)

    def test_ui_includes_live_gt_drawer_and_multiclass_query(self) -> None:
        html_src = (_SW_ROOT / "scripts" / "fics_final_demo.py").read_text(encoding="utf-8")
        js_src = (_SW_ROOT / "scripts" / "static_r28" / "r28_app.js").read_text(encoding="utf-8")
        self.assertIn("liveGtDrawer", html_src)
        self.assertIn("找电阻和电容", html_src)
        self.assertIn("预测框数（未核真值）", html_src)
        self.assertIn("局部已核窗口评分（非整板 F1）", js_src)
        self.assertIn("renderLiveGtResult", js_src)


if __name__ == "__main__":
    unittest.main()
