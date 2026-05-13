"""Download and display sample COCO instance segmentation annotations.

Prints a small set of image, category, and annotation records for quick dataset
inspection before running a full training or evaluation job.

Usage:
    python tools/sample_coco_data.py              # Check for existing data
    python tools/sample_coco_data.py --download   # Download if not present
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path


def download_coco_annotations(data_dir: Path):
    """Download COCO 2017 validation annotations if not present."""
    annotations_dir = data_dir / "coco" / "annotations"
    annotations_file = annotations_dir / "instances_val2017.json"

    if annotations_file.exists():
        print(f"Annotations already exist at {annotations_file}")
        return annotations_file

    annotations_dir.mkdir(parents=True, exist_ok=True)

    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = data_dir / "annotations_trainval2017.zip"

    print(f"Downloading COCO annotations from {url}...")
    print("(This is ~250MB, may take a few minutes)")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir / "coco")

    zip_path.unlink()
    print(f"Downloaded and extracted to {annotations_dir}")

    return annotations_file


def print_sample_records(annotations_file: Path, num_images: int = 5, num_annotations: int = 5):
    """Print sample records from COCO annotations."""

    print(f"\nLoading annotations from {annotations_file}...")
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    # Build category lookup
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Print dataset overview
    print("\n" + "="*70)
    print("COCO DATASET OVERVIEW")
    print("="*70)
    print(f"Total images: {len(coco_data['images']):,}")
    print(f"Total annotations: {len(coco_data['annotations']):,}")
    print(f"Total categories: {len(coco_data['categories'])}")

    # Print sample categories
    print("\n" + "-"*70)
    print("SAMPLE CATEGORIES (first 10)")
    print("-"*70)
    for cat in coco_data["categories"][:10]:
        print(f"  ID: {cat['id']:3d} | Name: {cat['name']:<15} | Supercategory: {cat['supercategory']}")

    # Print sample images
    print("\n" + "-"*70)
    print(f"SAMPLE IMAGES (first {num_images})")
    print("-"*70)
    for img in coco_data["images"][:num_images]:
        print(f"  ID: {img['id']:6d} | File: {img['file_name']:<25} | Size: {img['width']}x{img['height']}")

    # Print sample annotations
    print("\n" + "-"*70)
    print(f"SAMPLE ANNOTATIONS (first {num_annotations})")
    print("-"*70)
    for ann in coco_data["annotations"][:num_annotations]:
        cat_name = categories.get(ann["category_id"], "unknown")
        bbox = ann["bbox"]
        # Truncate segmentation for display
        segmentation = str(ann["segmentation"])
        seg_preview = segmentation[:50] + "..." if len(segmentation) > 50 else segmentation

        print(f"\n  Annotation ID: {ann['id']}")
        print(f"    Image ID:     {ann['image_id']}")
        print(f"    Category:     {cat_name} (id={ann['category_id']})")
        print(f"    Bounding Box: [x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}]")
        print(f"    Area:         {ann['area']:.1f} pixels")
        print(f"    Is Crowd:     {ann['iscrowd']}")
        print(f"    Segmentation: {seg_preview}")

    # Print raw JSON for a small inspectable sample.
    print("\n" + "="*70)
    print("RAW JSON FORMAT (first 3 annotations)")
    print("="*70)
    sample_output = {
        "images": coco_data["images"][:3],
        "annotations": coco_data["annotations"][:3],
        "categories": coco_data["categories"][:5],
    }
    print(json.dumps(sample_output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Display sample COCO annotation records")
    parser.add_argument("--download", action="store_true", help="Download annotations if not present")
    args = parser.parse_args()

    # Set data directory
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "datasets"

    print("COCO Instance Segmentation Data Sample")
    print("="*70)

    # Check for existing annotations
    possible_paths = [
        data_dir / "coco" / "annotations" / "instances_val2017.json",
        data_dir / "coco" / "annotations" / "instances_train2017.json",
        Path(os.environ.get("DETECTRON2_DATASETS", "")) / "coco" / "annotations" / "instances_val2017.json",
    ]

    annotations_file = None
    for path in possible_paths:
        if path.exists():
            annotations_file = path
            print(f"Found existing annotations: {path}")
            break

    if annotations_file is None:
        print("No local COCO annotations found.")
        if args.download:
            annotations_file = download_coco_annotations(data_dir)
        else:
            print("\nTo download automatically, run:")
            print("  python tools/sample_coco_data.py --download")
            print("\nOr manually download:")
            print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            print("  unzip annotations_trainval2017.zip -d datasets/coco/")
            return

    print_sample_records(annotations_file)


if __name__ == "__main__":
    main()
