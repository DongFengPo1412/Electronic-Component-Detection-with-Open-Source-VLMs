"""Unit tests for isolated FICS-PCB DSLR parsing, semantics, provenance, and board isolation."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcb_vlm.config import get_canonical_categories
from pcb_vlm.data.fics_dslr import (
    FICS_CANONICAL_CATEGORIES,
    audit_and_build_fics_dslr_bundle,
    parse_board_and_side,
    parse_component_location,
    validate_physical_board_disjoint_splits,
)


def _write_synthetic_s18_fixture(root: Path, rows: list[dict], size=(400, 300)) -> Path:
    board_dir = root / "s18"
    img_dir = board_dir / "DSLR" / "img"
    ann_dir = board_dir / "DSLR" / "annotation"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 20)).save(img_dir / "s18_front.tif", format="TIFF")
    with open(ann_dir / "s18_front.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "component_location", "component_type", "text_on_component", "logo"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return board_dir


class TestFicsDslrParser(unittest.TestCase):
    def test_namespace_isolated_from_wacv_11_classes(self):
        wacv_classes = get_canonical_categories()
        fics_classes = [c["name"] for c in FICS_CANONICAL_CATEGORIES]
        self.assertEqual(len(wacv_classes), 11)
        self.assertEqual(fics_classes, ["capacitor", "resistor", "inductor", "transistor", "diode", "ic"])

    def test_rect_and_polygon_unexpanded_exact_coordinates(self):
        raw_r, box_r, note_r = parse_component_location(
            '{"name":"rect","x":10,"y":20,"width":30,"height":40}'
        )
        self.assertEqual(raw_r["shape"], "rect")
        self.assertEqual(box_r, [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(note_r, "original_rect_unexpanded")

        raw_p, box_p, note_p = parse_component_location(
            '{"name":"polygon","all_points_x":[12,52,50],"all_points_y":[15,15,45]}'
        )
        self.assertEqual(raw_p["shape"], "polygon")
        self.assertEqual(box_p, [12.0, 15.0, 52.0, 45.0])
        self.assertEqual(note_p, "axis_aligned_envelope_from_polygon_unexpanded")

    def test_rejects_unknown_shape_and_non_json_code(self):
        with self.assertRaises(ValueError):
            parse_component_location('{"name":"circle","cx":10,"cy":10,"r":5}')
        with self.assertRaises(ValueError):
            parse_component_location("__import__('os').system('echo bad')")
        with self.assertRaises(ValueError):
            parse_component_location('{"name":"rect","x":0,"y":0,"width":0,"height":10}')
        # Collinear diagonal polygon has positive axis-aligned bbox but zero geometric area
        with self.assertRaises(ValueError):
            parse_component_location('{"name":"polygon","all_points_x":[0,10,20],"all_points_y":[0,10,20]}')

    def test_rejects_duplicate_image_stem_across_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            board_dir = _write_synthetic_s18_fixture(
                Path(tmp),
                rows=[{
                    "image_name": "s18_front.tif",
                    "component_location": '{"name":"rect","x":10,"y":20,"width":30,"height":20}',
                    "component_type": "capacitors",
                    "text_on_component": "",
                    "logo": "no",
                }],
            )
            Image.new("RGB", (400, 300), color=(20, 40, 20)).save(board_dir / "DSLR" / "img" / "s18_front.png")
            with self.assertRaises(ValueError):
                audit_and_build_fics_dslr_bundle(board_dir, strict=True)

    def test_bundle_preserves_raw_class_row_number_and_rejects_oob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board_dir = _write_synthetic_s18_fixture(
                root,
                rows=[
                    {
                        "image_name": "s18_front.tif",
                        "component_location": '{"name":"rect","x":10,"y":20,"width":50,"height":25}',
                        "component_type": "capacitors",
                        "text_on_component": "",
                        "logo": "no",
                    },
                    {
                        "image_name": "s18_front.tif",
                        "component_location": '{"name":"polygon","all_points_x":[100,150,150,100],"all_points_y":[50,50,100,100]}',
                        "component_type": "ICs",
                        "text_on_component": "CHIP-X",
                        "logo": "yes",
                    },
                ],
            )
            bundle = audit_and_build_fics_dslr_bundle(board_dir, strict=True)
            self.assertTrue(all(bundle["checks"].values()))
            self.assertEqual(bundle["raw_type_counts"], {"capacitors": 1, "ICs": 1})
            self.assertEqual(bundle["object_mappings"][0]["csv_row"], 2)
            self.assertEqual(bundle["object_mappings"][1]["csv_row"], 3)
            self.assertEqual(bundle["object_mappings"][1]["raw_component_type"], "ICs")
            self.assertEqual(bundle["coco"]["annotations"][0]["bbox"], [10.0, 20.0, 50.0, 25.0])

            # Verify exported COCO aligns cleanly with scripts/evaluate.py
            scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            import evaluate as eval_mod
            cat_name = {c["id"]: c["name"] for c in bundle["coco"]["categories"]}
            pred_payload = {
                "predictions": [{
                    "image_id": bundle["coco"]["images"][0]["id"],
                    "status": "success",
                    "objects": [
                        {"label": cat_name[a["category_id"]], "bbox_xyxy": a["bbox_xyxy"], "score": 1.0}
                        for a in bundle["coco"]["annotations"]
                    ],
                }]
            }
            ev = eval_mod.evaluate_detections(bundle["coco"], pred_payload, iou_threshold=0.50)
            self.assertEqual(ev["micro"]["f1"], 1.0)
            self.assertEqual(sum(c["tp"] for c in ev["per_class"].values()), 2)

            # Out-of-bounds must fail in strict mode and record anomaly in non-strict mode
            oob_dir = _write_synthetic_s18_fixture(
                root / "oob",
                rows=[
                    {
                        "image_name": "s18_front.tif",
                        "component_location": '{"name":"rect","x":380,"y":20,"width":50,"height":25}',
                        "component_type": "unknown_widget",
                        "text_on_component": "",
                        "logo": "no",
                    }
                ],
            )
            with self.assertRaises(ValueError):
                audit_and_build_fics_dslr_bundle(oob_dir, strict=True)
            non_strict = audit_and_build_fics_dslr_bundle(oob_dir, strict=False)
            self.assertEqual(len(non_strict["anomalies"]), 1)
            self.assertIn("unknown_component_type:unknown_widget", non_strict["anomalies"][0]["issues"])

    def test_same_board_sides_cannot_split_across_partitions(self):
        self.assertEqual(parse_board_and_side("s18_front"), ("s18", "front"))
        manifest = [
            {"image_id": "s18_front", "physical_board_id": "s18"},
            {"image_id": "s18_back", "physical_board_id": "s18"},
        ]
        with self.assertRaises(ValueError):
            validate_physical_board_disjoint_splits(
                {"train": ["s18_front"], "val": ["s18_back"]},
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
