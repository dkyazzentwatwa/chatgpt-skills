#!/usr/bin/env python3
"""
Image Comparison Tool - Compare images with SSIM.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class ImageComparisonTool:
    """Compare images."""

    def __init__(self):
        """Initialize tool."""
        self.img1 = None
        self.img2 = None

    def load_images(self, path1: str, path2: str) -> 'ImageComparisonTool':
        """Load two images."""
        self.img1 = cv2.imread(path1)
        self.img2 = cv2.imread(path2)

        # Resize if needed
        if self.img1.shape != self.img2.shape:
            h, w = min(self.img1.shape[0], self.img2.shape[0]), min(self.img1.shape[1], self.img2.shape[1])
            self.img1 = cv2.resize(self.img1, (w, h))
            self.img2 = cv2.resize(self.img2, (w, h))

        return self

    def calculate_ssim(self) -> float:
        """Calculate SSIM similarity score."""
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)
        self.diff_img = diff

        return score

    def get_difference_image(self) -> np.ndarray:
        """Get difference heatmap."""
        diff = cv2.absdiff(self.img1, self.img2)
        return diff

    def create_comparison(self, output: str) -> str:
        """Create side-by-side comparison."""
        score = self.calculate_ssim()
        diff = self.get_difference_image()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Image 1')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Image 2')
        axes[1].axis('off')

        axes[2].imshow(diff)
        axes[2].set_title(f'Difference (SSIM: {score:.3f})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output


def main():
    parser = argparse.ArgumentParser(description="Image Comparison Tool")

    parser.add_argument("--image1", required=True, help="First image")
    parser.add_argument("--image2", required=True, help="Second image")
    parser.add_argument("--output", "-o", required=True, help="Output comparison image")

    args = parser.parse_args()

    tool = ImageComparisonTool()
    tool.load_images(args.image1, args.image2)

    score = tool.calculate_ssim()
    print(f"SSIM Similarity Score: {score:.3f}")

    if score > 0.95:
        print("Images are very similar")
    elif score > 0.8:
        print("Images are similar")
    elif score > 0.5:
        print("Images are somewhat different")
    else:
        print("Images are very different")

    tool.create_comparison(args.output)
    print(f"\nComparison saved: {args.output}")


if __name__ == "__main__":
    main()
