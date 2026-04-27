import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image


class ImageMerger:
    def __init__(self, poly_edge, target, result_img, prompt_str, output_dir):
        self.poly_edge = poly_edge
        self.target = target
        self.result_img = result_img
        self.prompt_str = prompt_str
        self.output_dir = output_dir

    @classmethod
    def from_image_paths(cls, poly_path, target_path, result_path, prompt_str, output_dir):
        return cls(
            Image.open(poly_path),
            Image.open(target_path),
            Image.open(result_path),
            prompt_str,
            output_dir,
        )

    def merge_and_save(self, filename="merged.png"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(self.poly_edge)
        axes[0].set_title("PolyEdge")
        axes[0].axis("off")

        axes[1].imshow(self.target)
        axes[1].set_title("GroundTruth")
        axes[1].axis("off")

        axes[2].imshow(self.result_img)
        axes[2].set_title("Result")
        axes[2].axis("off")

        plt.tight_layout()
        fig.text(0.5, -0.05, f"prompt: {self.prompt_str}", ha="center", fontsize=18)

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2, facecolor="white")
        plt.close(fig)
        return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge three images into one panel.")
    parser.add_argument("--poly", type=str, required=True, help="PolyEdge image path")
    parser.add_argument("--target", type=str, required=True, help="Target image path")
    parser.add_argument("--result", type=str, required=True, help="Result image path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--output-dir", type=str, default="./merged_results", help="Output directory")
    parser.add_argument("--filename", type=str, default="merged.png", help="Output filename")
    args = parser.parse_args()

    merger = ImageMerger.from_image_paths(
        args.poly,
        args.target,
        args.result,
        args.prompt,
        args.output_dir,
    )
    saved_path = merger.merge_and_save(args.filename)
    print(f"[INFO] Merged image saved at: {saved_path}")
