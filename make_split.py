"""Build a scene-grouped train/valid split in data/grouped/ (the download is left untouched).

Why: the provided split puts near-identical frames of the same scene on both sides
(see METHODOLOGY.md), so its validation scores are optimistic. Here all 999 images are
clustered into scenes and each scene goes entirely to train or entirely to valid.

    python make_split.py              # materialize the committed split (splits/grouped_split.csv)
    python make_split.py --recompute  # re-cluster and rewrite the manifest

Scenes = average-linkage clusters of ImageNet ResNet-18 embeddings, cut at cosine
distance 0.25. The manifest is committed so the split is identical on every machine.
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "Sampled-YD-Object-Detection-2"
OUT = ROOT / "data" / "grouped"
MANIFEST = ROOT / "splits" / "grouped_split.csv"
DISTANCE_CUT = 0.25
VALID_FRACTION = 0.20
SEED = 0


def source_images():
    return sorted(p for split in ("train", "valid") for p in (SOURCE / split / "images").iterdir())


def label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def classes_in(image_path):
    return {int(line.split()[0]) for line in label_path(image_path).read_text().splitlines() if line.strip()}


def embed(images):
    import torch
    import torchvision
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in images[i : i + 64]]).to(device)
            chunks.append(torch.nn.functional.normalize(model(batch), dim=1).cpu())
    return torch.cat(chunks).numpy()


def recompute(images):
    from scipy.cluster.hierarchy import fcluster, linkage

    groups = fcluster(linkage(embed(images), "average", metric="cosine"), DISTANCE_CUT, "distance")
    members = defaultdict(list)
    for image, group in zip(images, groups):
        members[int(group)].append(image)

    # Whole scenes go to valid, in random order, until it holds ~20% of the images.
    order = sorted(members)
    random.Random(SEED).shuffle(order)
    target = VALID_FRACTION * len(images)
    valid_groups, n_valid = set(), 0
    for group in order:
        size = len(members[group])
        if n_valid + size <= target * 1.05:
            valid_groups.add(group)
            n_valid += size

    MANIFEST.parent.mkdir(exist_ok=True)
    with MANIFEST.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "scene", "split"])
        for image, group in zip(images, groups):
            writer.writerow([image.name, int(group), "valid" if int(group) in valid_groups else "train"])
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(members)} scenes")


def materialize(images):
    by_name = {p.name: p for p in images}
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    missing = [r["image"] for r in rows if r["image"] not in by_name]
    if missing or len(rows) != len(images):
        raise SystemExit("Manifest does not match the downloaded dataset; run with --recompute.")

    if OUT.exists():
        shutil.rmtree(OUT)
    stats = defaultdict(Counter)
    for row in rows:
        image = by_name[row["image"]]
        for kind, src in (("images", image), ("labels", label_path(image))):
            dst = OUT / row["split"] / kind
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        stats[row["split"]]["images"] += 1
        stats[row["split"]]["scenes"] = len({r["scene"] for r in rows if r["split"] == row["split"]})
        for line in label_path(image).read_text().splitlines():
            if line.strip():
                stats[row["split"]][f"class {line.split()[0]} boxes"] += 1
    for split in ("train", "valid"):
        print(split, dict(stats[split]))
    shared = {r["scene"] for r in rows if r["split"] == "train"} & {r["scene"] for r in rows if r["split"] == "valid"}
    print(f"scenes on both sides: {len(shared)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="re-cluster and rewrite the manifest")
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit("Dataset not found; run download_data.py first.")
    images = source_images()
    if args.recompute or not MANIFEST.exists():
        recompute(images)
    materialize(images)


if __name__ == "__main__":
    main()
