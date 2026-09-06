"""Prepare NIH ChestX-ray14 for the ImageFolder-based notebooks."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("nih_root", type=Path, help="Extracted NIH ChestX-ray14 root")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive") / "nih_imagefolder",
        help="ImageFolder output directory",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def read_names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def locate_images(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.rglob("*.png")}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    nih_root = args.nih_root.resolve()
    output = args.output.resolve()
    metadata_path = find_file(nih_root, "Data_Entry_2017.csv")
    train_val_path = find_file(nih_root, "train_val_list.txt")
    test_path = find_file(nih_root, "test_list.txt")
    image_paths = locate_images(nih_root)

    metadata = pd.read_csv(metadata_path)
    metadata["Image Index"] = metadata["Image Index"].astype(str)
    metadata["Patient ID"] = metadata["Patient ID"].astype(str)
    metadata["label"] = metadata["Finding Labels"].map(
        lambda value: 1 if "Pneumonia" in str(value).split("|") else 0 if str(value) == "No Finding" else -1
    )
    metadata = metadata[metadata["label"] >= 0].copy()

    official_train_val = read_names(train_val_path)
    official_test = read_names(test_path)
    metadata["split"] = metadata["Image Index"].map(
        lambda name: "test" if name in official_test else "train_val" if name in official_train_val else "unknown"
    )
    metadata = metadata[metadata["split"] != "unknown"]
    metadata = metadata[metadata["Image Index"].isin(image_paths)]

    train_val = metadata[metadata["split"] == "train_val"].copy()
    test = metadata[metadata["split"] == "test"].copy()
    rng = random.Random(args.seed)
    patients = list(train_val["Patient ID"].drop_duplicates())
    rng.shuffle(patients)
    val_patients = set(patients[: max(1, int(len(patients) * args.val_fraction))])
    train_val["split"] = train_val["Patient ID"].map(lambda patient: "val" if patient in val_patients else "train")
    test["split"] = "test"
    selected = pd.concat([train_val, test], ignore_index=True)

    for _, row in selected.iterrows():
        image_name = row["Image Index"]
        class_name = "PNEUMONIA" if row["label"] == 1 else "NORMAL"
        destination = output / row["split"] / class_name / image_name
        link_or_copy(image_paths[image_name], destination)

    selected.to_csv(output / "metadata_used.csv", index=False)
    print(f"Prepared {len(selected)} images at {output}")
    print(selected.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()