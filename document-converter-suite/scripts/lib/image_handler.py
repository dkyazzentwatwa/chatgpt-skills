#!/usr/bin/env python3
"""Image extraction and handling for document converter suite."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


@dataclass
class ImageRef:
    """Reference to an extracted image."""
    id: str  # Hash-based unique ID
    path: Path  # Path to saved image file
    format: str  # Image format (png, jpg, etc.)
    width: Optional[int] = None
    height: Optional[int] = None
    alt_text: Optional[str] = None


def save_image(
    image_data: bytes,
    output_dir: Path,
    prefix: str = "img",
    alt_text: Optional[str] = None
) -> ImageRef:
    """
    Save image data to file with hash-based deduplication.

    Args:
        image_data: Raw image bytes
        output_dir: Directory to save image
        prefix: Filename prefix (default: "img")
        alt_text: Optional alt text for the image

    Returns:
        ImageRef with image metadata

    Raises:
        ValueError: If Pillow is not available
        RuntimeError: If image format cannot be determined
    """
    if not PILLOW_AVAILABLE:
        raise ValueError("Pillow library is required for image extraction. Install with: pip install Pillow")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate hash ID to avoid duplicates
    image_hash = hashlib.sha256(image_data).hexdigest()[:12]
    image_id = f"{prefix}_{image_hash}"

    # Detect format using PIL
    try:
        from io import BytesIO
        img = Image.open(BytesIO(image_data))
        format_lower = img.format.lower() if img.format else "png"
        width, height = img.size

        # Save image
        image_path = output_dir / f"{image_id}.{format_lower}"

        # Check if already exists (deduplication)
        if not image_path.exists():
            with open(image_path, 'wb') as f:
                f.write(image_data)

        return ImageRef(
            id=image_id,
            path=image_path,
            format=format_lower,
            width=width,
            height=height,
            alt_text=alt_text
        )

    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")


def get_image_placeholder(image_ref: ImageRef) -> str:
    """
    Generate a text placeholder for an image.

    Args:
        image_ref: Image reference

    Returns:
        Formatted placeholder string
    """
    if image_ref.alt_text:
        return f"[Image: {image_ref.alt_text} ({image_ref.path.name})]"
    else:
        return f"[Image: {image_ref.path.name}]"
