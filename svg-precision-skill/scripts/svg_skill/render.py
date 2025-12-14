from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, Optional, Union


def render_png(svg: Union[str, bytes], out_png: str, *, scale: float = 1.0, from_file: bool = True) -> Dict[str, Any]:
    """Render an SVG to PNG using CairoSVG (optional dependency).

    Args:
        svg: path (if from_file) or SVG bytes/string
        out_png: output file path
        scale: render scale multiplier
        from_file: treat `svg` as file path when True
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        return {"ok": False, "error": "CairoSVG not available. Install with: pip install cairosvg", "detail": str(e)}

    try:
        if from_file:
            data = open(str(svg), "rb").read()
        else:
            data = svg if isinstance(svg, (bytes, bytearray)) else str(svg).encode("utf-8")

        cairosvg.svg2png(bytestring=data, write_to=out_png, scale=float(scale))
        return {"ok": True, "out_png": out_png}
    except Exception as e:
        return {"ok": False, "error": f"render failed: {e}"}


def diff_svgs(a_svg: str, b_svg: str, out_png: str, *, scale: float = 1.0) -> Dict[str, Any]:
    """Render two SVGs and output a pixel diff PNG (requires CairoSVG + Pillow)."""
    try:
        from PIL import Image, ImageChops  # type: ignore
    except Exception as e:
        return {"ok": False, "error": "Pillow not available. Install with: pip install pillow", "detail": str(e)}

    with tempfile.TemporaryDirectory() as td:
        a_png = os.path.join(td, "a.png")
        b_png = os.path.join(td, "b.png")
        ra = render_png(a_svg, a_png, scale=scale, from_file=True)
        if not ra.get("ok"):
            return ra
        rb = render_png(b_svg, b_png, scale=scale, from_file=True)
        if not rb.get("ok"):
            return rb

        im_a = Image.open(a_png).convert("RGBA")
        im_b = Image.open(b_png).convert("RGBA")

        # Pad to same size
        w = max(im_a.size[0], im_b.size[0])
        h = max(im_a.size[1], im_b.size[1])
        if im_a.size != (w, h):
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(im_a, (0, 0))
            im_a = tmp
        if im_b.size != (w, h):
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(im_b, (0, 0))
            im_b = tmp

        diff = ImageChops.difference(im_a, im_b)
        diff.save(out_png)

        bbox = diff.getbbox()
        changed = bbox is not None
        return {"ok": True, "out_png": out_png, "changed": changed, "bbox": bbox}
