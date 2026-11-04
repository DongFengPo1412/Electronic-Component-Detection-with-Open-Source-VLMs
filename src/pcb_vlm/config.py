"""Canonical configuration and category management for PCB-VLM."""

import os
import yaml
from typing import List, Optional

CANONICAL_PCB_CATEGORIES: List[str] = [
    "button", "capacitor", "clock", "connector", "diode",
    "ic", "inductor", "led", "resistor", "switch", "transistor"
]

CANONICAL_CATEGORY_COCO_LIST = [
    {"id": 1, "name": "capacitor"},
    {"id": 2, "name": "resistor"},
    {"id": 3, "name": "ic"},
    {"id": 4, "name": "connector"},
    {"id": 5, "name": "led"},
    {"id": 6, "name": "transistor"},
    {"id": 7, "name": "diode"},
    {"id": 8, "name": "inductor"},
    {"id": 9, "name": "button"},
    {"id": 10, "name": "switch"},
    {"id": 11, "name": "clock"},
]


def get_canonical_categories(classes_yaml_path: Optional[str] = None) -> List[str]:
    """Retrieve the sorted canonical category list from classes.yaml or fallback."""
    if classes_yaml_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        candidate = os.path.join(base_dir, "configs", "classes.yaml")
        if os.path.exists(candidate):
            classes_yaml_path = candidate

    if classes_yaml_path and os.path.exists(classes_yaml_path):
        with open(classes_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "categories" in data and isinstance(data["categories"], list):
            cats = [c["name"] for c in data["categories"] if "name" in c]
            if cats:
                validate_category_consistency(cats)
                return sorted(cats)

    return list(CANONICAL_PCB_CATEGORIES)


def validate_category_consistency(categories: List[str], expected: Optional[List[str]] = None) -> None:
    """Verify that categories match the canonical 11 classes, raising ValueError if any are missing."""
    if expected is None:
        expected = CANONICAL_PCB_CATEGORIES

    cat_set = set(categories)
    expected_set = set(expected)

    missing = expected_set - cat_set
    if missing or cat_set != expected_set or len(categories) != len(cat_set):
        raise ValueError(
            f"Category inconsistency detected! Missing required canonical categories: {sorted(missing)}. "
            f"Provided: {sorted(categories)}, Expected: {sorted(expected)}"
        )


def get_category_definitions():
    from pathlib import Path
    data = yaml.safe_load((Path(__file__).resolve().parents[2] / 'configs/classes.yaml').read_text(encoding='utf-8'))['categories']
    validate_category_consistency([c['name'] for c in data])
    if len({c['id'] for c in data}) != len(data):
        raise ValueError('Duplicate category IDs')
    return data

CANONICAL_CATEGORY_COCO_LIST = [{"id": c["id"], "name": c["name"]} for c in get_category_definitions()]
