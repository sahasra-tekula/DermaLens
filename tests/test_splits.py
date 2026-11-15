from pathlib import Path

import pandas as pd

from src.data_pipeline import CLASS_TO_IDX, CLASSES, PROJECT_ROOT, SPLIT_DIR, assert_no_lesion_overlap


def test_fixed_label_mapping_is_complete_and_bijective():
    assert tuple(CLASS_TO_IDX) == CLASSES
    assert set(CLASS_TO_IDX.values()) == set(range(7))


def test_saved_splits_have_no_lesion_overlap():
    frames = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    assert all({"image_id", "lesion_id", "dx"} <= set(frame.columns) for frame in frames.values())
    assert_no_lesion_overlap(frames)
    assert all(set(frame["dx"]) == set(CLASSES) for frame in frames.values())
    ids = pd.concat(list(frames.values()))["image_id"]
    assert not ids.duplicated().any()


def test_exact_duplicate_copies_are_excluded_from_manifest():
    duplicate_report = pd.read_csv(PROJECT_ROOT / "reports" / "exact_duplicate_images.csv")
    split_ids = set()
    for name in ("train", "val", "test"):
        split_ids.update(pd.read_csv(SPLIT_DIR / f"{name}.csv")["image_id"])
    assert len(duplicate_report) == 2
    assert set(duplicate_report["excluded_image_id"]).isdisjoint(split_ids)
    assert set(duplicate_report["kept_image_id"]).issubset(split_ids)
