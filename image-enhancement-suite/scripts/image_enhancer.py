#!/usr/bin/env python3
"""
Image Enhancement Suite - Professional image processing toolkit
Batch resize, crop, watermark, color correct, convert, and compress images.
"""

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# Try to import OpenCV for advanced features
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageError(Exception):
    """Custom exception for image processing errors."""
    pass


@dataclass
class EnhancerConfig:
    """Configuration for image processing."""
    default_quality: int = 85
    default_format: str = 'jpeg'
    preserve_metadata: bool = False
    color_profile: str = 'sRGB'
    dpi: int = 72
    max_dimension: int = 10000
    watermark_font: str = 'arial.ttf'
    watermark_fallback_font: str = 'DejaVuSans.ttf'

    def update(self, settings: Dict[str, Any]) -> None:
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Quality presets for different use cases
PRESETS = {
    'web': {'max_width': 1200, 'quality': 85, 'format': 'webp'},
    'thumbnail': {'width': 150, 'height': 150, 'crop': True, 'quality': 80},
    'preview': {'max_width': 400, 'quality': 70},
    'instagram': {'width': 1080, 'height': 1080, 'crop': True, 'quality': 90},
    'instagram_portrait': {'width': 1080, 'height': 1350, 'crop': True, 'quality': 90},
    'instagram_landscape': {'width': 1080, 'height': 608, 'crop': True, 'quality': 90},
    'twitter': {'width': 1200, 'height': 675, 'crop': True, 'quality': 85},
    'facebook': {'width': 1200, 'height': 630, 'crop': True, 'quality': 85},
    'linkedin': {'width': 1200, 'height': 627, 'crop': True, 'quality': 85},
    'print_4x6': {'width': 1800, 'height': 1200, 'dpi': 300, 'quality': 95},
    'print_8x10': {'width': 3000, 'height': 2400, 'dpi': 300, 'quality': 95},
    'print_11x14': {'width': 4200, 'height': 3300, 'dpi': 300, 'quality': 95},
    'hd': {'max_width': 1920, 'max_height': 1080, 'quality': 90},
    '4k': {'max_width': 3840, 'max_height': 2160, 'quality': 90},
}


# Position mappings for watermarks
POSITIONS = {
    'center': (0.5, 0.5),
    'top-left': (0.05, 0.05),
    'top-right': (0.95, 0.05),
    'bottom-left': (0.05, 0.95),
    'bottom-right': (0.95, 0.95),
    'top-center': (0.5, 0.05),
    'bottom-center': (0.5, 0.95),
}


class ImageEnhancer:
    """
    Main class for image enhancement and processing.

    Supports chaining operations for fluent API:
        ImageEnhancer("photo.jpg").resize(800).sharpen(0.5).save("output.jpg")
    """

    def __init__(self, source: Union[str, Path, Image.Image, bytes]):
        """
        Initialize with image source.

        Args:
            source: File path, PIL Image, or bytes
        """
        self.config = EnhancerConfig()
        self._original: Optional[Image.Image] = None
        self._history: List[str] = []

        if isinstance(source, (str, Path)):
            self._path = Path(source)
            self._load_file(self._path)
        elif isinstance(source, Image.Image):
            self._path = None
            self._img = source.copy()
            self._original = source.copy()
        elif isinstance(source, bytes):
            self._path = None
            self._img = Image.open(io.BytesIO(source))
            self._original = self._img.copy()
        else:
            raise ImageError(f"Unsupported source type: {type(source)}")

        # Ensure RGB mode for most operations
        if self._img.mode == 'RGBA':
            pass  # Keep alpha
        elif self._img.mode != 'RGB':
            self._img = self._img.convert('RGB')

    def _load_file(self, path: Path) -> None:
        """Load image from file."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            self._img = Image.open(path)
            self._img.load()  # Force load
            self._original = self._img.copy()
        except Exception as e:
            raise ImageError(f"Failed to load image: {e}")

    @property
    def size(self) -> Tuple[int, int]:
        """Get current image size (width, height)."""
        return self._img.size

    @property
    def width(self) -> int:
        return self._img.size[0]

    @property
    def height(self) -> int:
        return self._img.size[1]

    @property
    def mode(self) -> str:
        return self._img.mode

    @property
    def format(self) -> Optional[str]:
        return self._img.format

    def reset(self) -> 'ImageEnhancer':
        """Reset to original image."""
        if self._original:
            self._img = self._original.copy()
            self._history = []
        return self

    def copy(self) -> 'ImageEnhancer':
        """Create a copy of the enhancer."""
        new = ImageEnhancer(self._img.copy())
        new._original = self._original.copy() if self._original else None
        new._history = self._history.copy()
        new.config = self.config
        return new

    # ==================== RESIZE OPERATIONS ====================

    def resize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        scale: Optional[float] = None,
        maintain_aspect: bool = True,
        resample: int = Image.Resampling.LANCZOS
    ) -> 'ImageEnhancer':
        """
        Resize the image.

        Args:
            width: Target width
            height: Target height
            max_width: Maximum width (fit within)
            max_height: Maximum height (fit within)
            scale: Scale factor (0.5 = 50%)
            maintain_aspect: Maintain aspect ratio
            resample: Resampling filter
        """
        current_w, current_h = self._img.size

        if scale is not None:
            new_w = int(current_w * scale)
            new_h = int(current_h * scale)
        elif max_width or max_height:
            # Fit within bounds
            new_w, new_h = current_w, current_h
            if max_width and current_w > max_width:
                ratio = max_width / current_w
                new_w = max_width
                new_h = int(current_h * ratio)
            if max_height and new_h > max_height:
                ratio = max_height / new_h
                new_h = max_height
                new_w = int(new_w * ratio)
        elif width and height and not maintain_aspect:
            new_w, new_h = width, height
        elif width:
            ratio = width / current_w
            new_w = width
            new_h = int(current_h * ratio) if maintain_aspect else (height or current_h)
        elif height:
            ratio = height / current_h
            new_h = height
            new_w = int(current_w * ratio) if maintain_aspect else (width or current_w)
        else:
            return self  # No resize needed

        self._img = self._img.resize((new_w, new_h), resample)
        self._history.append(f"resize({new_w}x{new_h})")
        return self

    def crop(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        position: str = 'center',
        box: Optional[Tuple[int, int, int, int]] = None
    ) -> 'ImageEnhancer':
        """
        Crop the image.

        Args:
            width: Crop width
            height: Crop height
            position: Crop position ('center', 'top-left', etc.)
            box: Explicit crop box (left, top, right, bottom)
        """
        if box:
            self._img = self._img.crop(box)
            self._history.append(f"crop(box={box})")
            return self

        if not width or not height:
            return self

        current_w, current_h = self._img.size

        # Calculate crop position
        if position in POSITIONS:
            cx, cy = POSITIONS[position]
        else:
            cx, cy = 0.5, 0.5

        # Calculate crop box
        left = int((current_w - width) * cx)
        top = int((current_h - height) * cy)
        right = left + width
        bottom = top + height

        # Ensure within bounds
        left = max(0, left)
        top = max(0, top)
        right = min(current_w, right)
        bottom = min(current_h, bottom)

        self._img = self._img.crop((left, top, right, bottom))
        self._history.append(f"crop({width}x{height}, {position})")
        return self

    def smart_crop(self, width: int, height: int) -> 'ImageEnhancer':
        """
        Smart crop that attempts to keep important content.
        Falls back to center crop if OpenCV not available.
        """
        if not HAS_CV2:
            return self.crop(width, height, 'center')

        # Convert to OpenCV format
        img_array = np.array(self._img)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Detect edges to find areas of interest
        edges = cv2.Canny(gray, 50, 150)

        # Find center of mass of edges
        moments = cv2.moments(edges)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx = self.width // 2
            cy = self.height // 2

        # Calculate crop box centered on detected center
        left = max(0, cx - width // 2)
        top = max(0, cy - height // 2)

        # Adjust if crop would extend beyond image
        if left + width > self.width:
            left = self.width - width
        if top + height > self.height:
            top = self.height - height

        left = max(0, left)
        top = max(0, top)

        self._img = self._img.crop((left, top, left + width, top + height))
        self._history.append(f"smart_crop({width}x{height})")
        return self

    # ==================== WATERMARK OPERATIONS ====================

    def watermark(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        position: str = 'bottom-right',
        opacity: float = 0.5,
        font_size: int = 24,
        color: str = 'white',
        scale: float = 0.2,
        tiled: bool = False,
        rotation: float = 0,
        padding: int = 10
    ) -> 'ImageEnhancer':
        """
        Add watermark to image.

        Args:
            text: Text watermark
            image: Image watermark (path or PIL Image)
            position: Position on image
            opacity: Watermark opacity (0-1)
            font_size: Font size for text
            color: Text color
            scale: Scale for image watermark (relative to main image)
            tiled: Tile watermark across image
            rotation: Rotation angle for text
            padding: Padding from edge
        """
        # Ensure we have RGBA mode
        if self._img.mode != 'RGBA':
            self._img = self._img.convert('RGBA')

        if text:
            self._add_text_watermark(
                text, position, opacity, font_size, color, tiled, rotation, padding
            )
        elif image:
            self._add_image_watermark(image, position, opacity, scale, padding)

        return self

    def _add_text_watermark(
        self,
        text: str,
        position: str,
        opacity: float,
        font_size: int,
        color: str,
        tiled: bool,
        rotation: float,
        padding: int
    ) -> None:
        """Add text watermark."""
        # Create text layer
        txt_layer = Image.new('RGBA', self._img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Try to load font
        try:
            font = ImageFont.truetype(self.config.watermark_font, font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype(self.config.watermark_fallback_font, font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Calculate opacity
        alpha = int(255 * opacity)

        # Parse color
        if isinstance(color, str):
            if color.lower() == 'white':
                fill = (255, 255, 255, alpha)
            elif color.lower() == 'black':
                fill = (0, 0, 0, alpha)
            else:
                fill = (255, 255, 255, alpha)
        else:
            fill = (*color[:3], alpha)

        if tiled:
            # Tile watermark
            step_x = text_w + 50
            step_y = text_h + 50
            for y in range(-text_h, self.height + text_h, step_y):
                for x in range(-text_w, self.width + text_w, step_x):
                    draw.text((x, y), text, font=font, fill=fill)
        else:
            # Single watermark
            if position in POSITIONS:
                px, py = POSITIONS[position]
                x = int((self.width - text_w) * px)
                y = int((self.height - text_h) * py)
            else:
                x = self.width - text_w - padding
                y = self.height - text_h - padding

            draw.text((x, y), text, font=font, fill=fill)

        # Apply rotation if needed
        if rotation != 0:
            txt_layer = txt_layer.rotate(rotation, expand=False, center=(self.width // 2, self.height // 2))

        # Composite
        self._img = Image.alpha_composite(self._img, txt_layer)
        self._history.append(f"watermark(text='{text[:20]}...')")

    def _add_image_watermark(
        self,
        watermark: Union[str, Path, Image.Image],
        position: str,
        opacity: float,
        scale: float,
        padding: int
    ) -> None:
        """Add image watermark."""
        # Load watermark image
        if isinstance(watermark, (str, Path)):
            wm = Image.open(watermark)
        else:
            wm = watermark.copy()

        # Ensure RGBA
        if wm.mode != 'RGBA':
            wm = wm.convert('RGBA')

        # Scale watermark
        wm_w = int(self.width * scale)
        wm_h = int(wm.height * (wm_w / wm.width))
        wm = wm.resize((wm_w, wm_h), Image.Resampling.LANCZOS)

        # Apply opacity
        alpha = wm.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        wm.putalpha(alpha)

        # Calculate position
        if position in POSITIONS:
            px, py = POSITIONS[position]
            x = int((self.width - wm_w) * px)
            y = int((self.height - wm_h) * py)
        else:
            x = self.width - wm_w - padding
            y = self.height - wm_h - padding

        # Composite
        self._img.paste(wm, (x, y), wm)
        self._history.append(f"watermark(image, scale={scale})")

    # ==================== COLOR ADJUSTMENTS ====================

    def brightness(self, factor: float) -> 'ImageEnhancer':
        """
        Adjust brightness.

        Args:
            factor: -1.0 to 1.0 (0 = no change)
        """
        enhancer = ImageEnhance.Brightness(self._img)
        self._img = enhancer.enhance(1 + factor)
        self._history.append(f"brightness({factor})")
        return self

    def contrast(self, factor: float) -> 'ImageEnhancer':
        """Adjust contrast. Factor: -1.0 to 1.0"""
        enhancer = ImageEnhance.Contrast(self._img)
        self._img = enhancer.enhance(1 + factor)
        self._history.append(f"contrast({factor})")
        return self

    def saturation(self, factor: float) -> 'ImageEnhancer':
        """Adjust saturation. Factor: -1.0 to 1.0"""
        enhancer = ImageEnhance.Color(self._img)
        self._img = enhancer.enhance(1 + factor)
        self._history.append(f"saturation({factor})")
        return self

    def sharpen(self, factor: float = 0.5) -> 'ImageEnhancer':
        """Sharpen image. Factor: 0 to 2.0"""
        enhancer = ImageEnhance.Sharpness(self._img)
        self._img = enhancer.enhance(1 + factor)
        self._history.append(f"sharpen({factor})")
        return self

    def adjust(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        sharpen: float = 0
    ) -> 'ImageEnhancer':
        """Apply multiple adjustments at once."""
        if brightness:
            self.brightness(brightness)
        if contrast:
            self.contrast(contrast)
        if saturation:
            self.saturation(saturation)
        if sharpen:
            self.sharpen(sharpen)
        return self

    def auto_enhance(self) -> 'ImageEnhancer':
        """Automatically enhance the image."""
        # Auto contrast
        self._img = ImageOps.autocontrast(self._img, cutoff=1)

        # Slight sharpening
        self.sharpen(0.3)

        self._history.append("auto_enhance()")
        return self

    # ==================== FILTERS ====================

    def filter(self, filter_name: str) -> 'ImageEnhancer':
        """
        Apply a preset filter.

        Available: grayscale, sepia, vintage, blur, sharpen, edge_enhance, emboss
        """
        filter_name = filter_name.lower()

        if filter_name == 'grayscale':
            self._img = self._img.convert('L').convert('RGB')
        elif filter_name == 'sepia':
            self._apply_sepia()
        elif filter_name == 'vintage':
            self._apply_vintage()
        elif filter_name == 'blur':
            self._img = self._img.filter(ImageFilter.BLUR)
        elif filter_name == 'sharpen':
            self._img = self._img.filter(ImageFilter.SHARPEN)
        elif filter_name == 'edge_enhance':
            self._img = self._img.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_name == 'emboss':
            self._img = self._img.filter(ImageFilter.EMBOSS)
        else:
            raise ImageError(f"Unknown filter: {filter_name}")

        self._history.append(f"filter({filter_name})")
        return self

    def _apply_sepia(self) -> None:
        """Apply sepia tone."""
        if self._img.mode != 'RGB':
            self._img = self._img.convert('RGB')

        pixels = np.array(self._img)
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b

        pixels[:, :, 0] = np.clip(tr, 0, 255)
        pixels[:, :, 1] = np.clip(tg, 0, 255)
        pixels[:, :, 2] = np.clip(tb, 0, 255)

        self._img = Image.fromarray(pixels.astype('uint8'))

    def _apply_vintage(self) -> None:
        """Apply vintage effect."""
        self._apply_sepia()
        self.contrast(-0.1)
        self.brightness(-0.05)

    def blur(self, radius: int = 2) -> 'ImageEnhancer':
        """Apply box blur."""
        self._img = self._img.filter(ImageFilter.BoxBlur(radius))
        self._history.append(f"blur({radius})")
        return self

    def gaussian_blur(self, radius: float = 2) -> 'ImageEnhancer':
        """Apply Gaussian blur."""
        self._img = self._img.filter(ImageFilter.GaussianBlur(radius))
        self._history.append(f"gaussian_blur({radius})")
        return self

    # ==================== FORMAT & COMPRESSION ====================

    def convert(self, format: str) -> 'ImageEnhancer':
        """Convert to specified format."""
        self._target_format = format.upper()
        self._history.append(f"convert({format})")
        return self

    def compress(self, quality: int = 85) -> 'ImageEnhancer':
        """Set compression quality."""
        self._quality = quality
        self._history.append(f"compress({quality})")
        return self

    def compress_to_size(self, max_kb: int) -> 'ImageEnhancer':
        """Compress to target file size."""
        quality = 95
        while quality > 10:
            buffer = io.BytesIO()
            self._img.save(buffer, format='JPEG', quality=quality)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_kb:
                break
            quality -= 5

        self._quality = quality
        self._history.append(f"compress_to_size({max_kb}kb, q={quality})")
        return self

    def optimize_png(self) -> 'ImageEnhancer':
        """Optimize PNG for smaller file size."""
        self._png_optimize = True
        self._history.append("optimize_png()")
        return self

    # ==================== PRESETS ====================

    def preset(self, preset_name: str) -> 'ImageEnhancer':
        """Apply a named preset."""
        if preset_name not in PRESETS:
            raise ImageError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

        settings = PRESETS[preset_name]

        if settings.get('crop'):
            self.smart_crop(settings['width'], settings['height'])
        elif 'width' in settings or 'height' in settings:
            self.resize(
                width=settings.get('width'),
                height=settings.get('height'),
                max_width=settings.get('max_width'),
                max_height=settings.get('max_height')
            )

        if 'quality' in settings:
            self._quality = settings['quality']

        if 'format' in settings:
            self._target_format = settings['format'].upper()

        self._history.append(f"preset({preset_name})")
        return self

    # ==================== METADATA ====================

    def get_metadata(self) -> Dict[str, Any]:
        """Extract image metadata."""
        metadata = {
            'size': self._img.size,
            'mode': self._img.mode,
            'format': self._img.format,
        }

        # Try to get EXIF data
        exif = self._img.getexif() if hasattr(self._img, 'getexif') else {}
        if exif:
            metadata['exif'] = dict(exif)

        return metadata

    def strip_metadata(self, keep: Optional[List[str]] = None) -> 'ImageEnhancer':
        """Remove metadata from image."""
        # Create new image without metadata
        data = list(self._img.getdata())
        new_img = Image.new(self._img.mode, self._img.size)
        new_img.putdata(data)
        self._img = new_img
        self._history.append("strip_metadata()")
        return self

    # ==================== SAVE/EXPORT ====================

    def save(
        self,
        path: Union[str, Path],
        quality: Optional[int] = None,
        format: Optional[str] = None
    ) -> str:
        """
        Save the processed image.

        Args:
            path: Output file path
            quality: JPEG/WebP quality (1-100)
            format: Output format (auto-detected from extension if not specified)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format
        if format:
            fmt = format.upper()
        elif hasattr(self, '_target_format'):
            fmt = self._target_format
        else:
            fmt = path.suffix.lstrip('.').upper()
            if not fmt:
                fmt = 'JPEG'

        # Map common format names
        fmt_map = {'JPG': 'JPEG', 'WEBP': 'WEBP', 'PNG': 'PNG', 'GIF': 'GIF'}
        fmt = fmt_map.get(fmt, fmt)

        # Determine quality
        q = quality or getattr(self, '_quality', self.config.default_quality)

        # Prepare image for saving
        save_img = self._img
        if fmt == 'JPEG' and save_img.mode == 'RGBA':
            # Remove alpha for JPEG
            background = Image.new('RGB', save_img.size, (255, 255, 255))
            background.paste(save_img, mask=save_img.split()[3])
            save_img = background
        elif fmt == 'JPEG' and save_img.mode != 'RGB':
            save_img = save_img.convert('RGB')

        # Save options
        save_kwargs = {}
        if fmt in ['JPEG', 'WEBP']:
            save_kwargs['quality'] = q
        if fmt == 'PNG' and getattr(self, '_png_optimize', False):
            save_kwargs['optimize'] = True

        save_img.save(path, format=fmt, **save_kwargs)
        return str(path)

    def to_bytes(self, format: str = 'JPEG', quality: int = 85) -> bytes:
        """Export image as bytes."""
        buffer = io.BytesIO()
        self.save(buffer, format=format, quality=quality)
        return buffer.getvalue()

    def __repr__(self) -> str:
        return f"ImageEnhancer({self.width}x{self.height}, {self.mode})"


# ==================== BATCH PROCESSING ====================

def batch_process(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    operations: List[Tuple[str, Dict[str, Any]]],
    formats: Optional[List[str]] = None,
    recursive: bool = False,
    rename_pattern: Optional[str] = None,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Process multiple images with the same operations.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        operations: List of (operation_name, kwargs) tuples
        formats: File formats to process (default: all image formats)
        recursive: Include subdirectories
        rename_pattern: Rename pattern (e.g., "{name}_processed_{index:03d}")
        parallel: Use parallel processing

    Returns:
        Results dict with success/failed counts
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default formats
    if formats is None:
        formats = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff']

    # Find files
    patterns = [f"*.{fmt}" for fmt in formats]
    if recursive:
        files = []
        for pattern in patterns:
            files.extend(input_dir.rglob(pattern))
    else:
        files = []
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))

    results = {'success': 0, 'failed': 0, 'errors': []}

    for idx, filepath in enumerate(sorted(files)):
        try:
            enhancer = ImageEnhancer(filepath)

            # Apply operations
            for op_name, kwargs in operations:
                method = getattr(enhancer, op_name, None)
                if method and callable(method):
                    method(**kwargs)

            # Determine output path
            if rename_pattern:
                new_name = rename_pattern.format(
                    name=filepath.stem,
                    index=idx,
                    ext=filepath.suffix.lstrip('.')
                )
                out_path = output_dir / f"{new_name}{filepath.suffix}"
            else:
                out_path = output_dir / filepath.name

            enhancer.save(out_path)
            results['success'] += 1

        except Exception as e:
            results['failed'] += 1
            results['errors'].append({'file': str(filepath), 'error': str(e)})

    return results


def generate_sizes(
    source: Union[str, Path],
    output_dir: Union[str, Path],
    widths: List[int],
    format: str = 'jpeg'
) -> List[str]:
    """
    Generate multiple sizes from a single image.

    Args:
        source: Source image
        output_dir: Output directory
        widths: List of widths to generate
        format: Output format

    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)

    generated = []
    enhancer = ImageEnhancer(source)

    for width in widths:
        new_enhancer = enhancer.copy()
        new_enhancer.resize(width=width)
        out_path = output_dir / f"{source_path.stem}_{width}.{format}"
        new_enhancer.save(out_path)
        generated.append(str(out_path))

    return generated


def generate_icons(
    source: Union[str, Path],
    output_dir: Union[str, Path],
    sizes: Optional[List[int]] = None
) -> List[str]:
    """
    Generate icon sizes from a single image.

    Args:
        source: Source image (should be square)
        output_dir: Output directory
        sizes: Icon sizes (default: standard favicon/app icon sizes)

    Returns:
        List of generated file paths
    """
    if sizes is None:
        sizes = [16, 32, 48, 64, 128, 180, 192, 256, 512]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)

    generated = []
    enhancer = ImageEnhancer(source)

    for size in sizes:
        new_enhancer = enhancer.copy()
        new_enhancer.resize(width=size, height=size, maintain_aspect=False)
        out_path = output_dir / f"{source_path.stem}_{size}x{size}.png"
        new_enhancer.save(out_path)
        generated.append(str(out_path))

    return generated


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Enhancement Suite')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('--resize', type=int, help='Resize to width')
    parser.add_argument('--quality', type=int, default=85, help='JPEG quality')
    parser.add_argument('--watermark', help='Add text watermark')
    parser.add_argument('--preset', help='Apply preset')
    parser.add_argument('--filter', help='Apply filter')
    parser.add_argument('--icons', help='Generate icons to directory')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.icons:
        # Generate icons mode
        icons = generate_icons(input_path, args.icons)
        print(f"Generated {len(icons)} icons")
    elif input_path.is_dir():
        # Batch mode
        operations = []
        if args.resize:
            operations.append(('resize', {'width': args.resize}))
        if args.watermark:
            operations.append(('watermark', {'text': args.watermark}))
        if args.preset:
            operations.append(('preset', {'preset_name': args.preset}))
        if args.filter:
            operations.append(('filter', {'filter_name': args.filter}))

        output_dir = args.output or 'processed'
        results = batch_process(input_path, output_dir, operations)
        print(f"Processed {results['success']} images, {results['failed']} failed")
    else:
        # Single file mode
        enhancer = ImageEnhancer(input_path)

        if args.resize:
            enhancer.resize(width=args.resize)
        if args.watermark:
            enhancer.watermark(text=args.watermark)
        if args.preset:
            enhancer.preset(args.preset)
        if args.filter:
            enhancer.filter(args.filter)

        output_path = args.output or input_path.with_stem(f"{input_path.stem}_enhanced")
        enhancer.save(output_path, quality=args.quality)
        print(f"Saved to: {output_path}")
