"""
Stitch multiple sample grids into a comparison figure.

Example (E1/E3 comparison):

python tools/grids/make_sample_grid_comparison.py \
  docs/assets/e1/e1_samples/step_10000.png \
  docs/assets/e1/e1_samples/step_50000.png \
  docs/assets/e3/e3_samples/step_5000.png \
  docs/assets/e3/e3_samples/step_10000.png \
  --titles "E1 @ 10k" "E1 @ 50k" "E3 @ 5k" "E3 @ 10k" \
  --out docs/assets/e3/e3_plots/e1_e3_samples_comparison.png
"""

import argparse
import math
import os
from typing import List

import matplotlib.pyplot as plt
from PIL import Image

import argparse
import math
import os
from typing import List

import matplotlib.pyplot as plt
from PIL import Image


def make_grid(
    image_paths: List[str],
    titles: List[str],
    out_path: str,
) -> None:
    if len(titles) == 0:
        titles = ["" for _ in image_paths]
    elif len(titles) != len(image_paths):
        raise ValueError("Number of titles must be 0 or equal to number of images.")

    n = len(image_paths)
    if n == 0:
        print("[sample_grid] No images provided, skipping.")
        return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                img_path = image_paths[idx]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis("off")
                if titles[idx]:
                    ax.set_title(titles[idx], fontsize=10)
            else:
                ax.axis("off")
            idx += 1

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[sample_grid] Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch multiple sample grids into a comparison figure."
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="Paths to sample grid images (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--titles",
        type=str,
        nargs="*",
        default=[],
        help="Optional titles for each image, in the same order.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the combined figure.",
    )
    args = parser.parse_args()
    make_grid(args.images, args.titles, args.out)


if __name__ == "__main__":
    main()