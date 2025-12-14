from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from xml.etree.ElementTree import Element, SubElement, tostring


_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"


def build_svg(spec: Dict[str, Any]) -> str:
    """Build an SVG string from a spec dict.

    The spec schema is documented in references/spec.md.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")

    canvas = spec.get("canvas") or {}
    width = canvas.get("width")
    height = canvas.get("height")
    view_box = canvas.get("viewBox")
    units = canvas.get("units", "px")

    if view_box is None:
        raise ValueError("canvas.viewBox is required")
    _assert_viewbox(view_box)

    # width/height optional but strongly recommended; default to viewBox dims
    if width is None or height is None:
        vb = [float(x) for x in str(view_box).split()]
        width = width or vb[2]
        height = height or vb[3]

    root = Element(
        "svg",
        {
            "xmlns": _SVG_NS,
            "xmlns:xlink": _XLINK_NS,
            "width": _fmt_len(width, units),
            "height": _fmt_len(height, units),
            "viewBox": str(view_box),
            "version": "1.1",
        },
    )

    # Optional metadata
    metadata = spec.get("metadata") or {}
    if isinstance(metadata, dict):
        title = metadata.get("title")
        desc = metadata.get("desc")
        if title:
            SubElement(root, "title").text = str(title)
        if desc:
            SubElement(root, "desc").text = str(desc)

    # Optional background
    background = canvas.get("background")
    if background:
        vb = [float(x) for x in str(view_box).split()]
        SubElement(
            root,
            "rect",
            {
                "x": _fmt_num(vb[0]),
                "y": _fmt_num(vb[1]),
                "width": _fmt_num(vb[2]),
                "height": _fmt_num(vb[3]),
                "fill": str(background),
            },
        )

    # defs
    defs_spec = spec.get("defs") or {}
    if isinstance(defs_spec, dict) and defs_spec:
        defs_el = SubElement(root, "defs")
        _emit_defs(defs_el, defs_spec)

    # elements
    elements = spec.get("elements")
    if not isinstance(elements, list):
        raise ValueError("elements must be a list")

    for node in elements:
        _emit_node(root, node)

    # Pretty-ish output (ElementTree doesn't pretty print reliably without minidom)
    svg_bytes = tostring(root, encoding="utf-8", xml_declaration=True)
    return svg_bytes.decode("utf-8")


def _emit_defs(defs_el: Element, defs_spec: Dict[str, Any]) -> None:
    for marker in defs_spec.get("markers") or []:
        _emit_marker(defs_el, marker)
    for grad in defs_spec.get("gradients") or []:
        _emit_gradient(defs_el, grad)
    for cp in defs_spec.get("clipPaths") or []:
        _emit_clip_path(defs_el, cp)


def _emit_marker(parent: Element, marker: Dict[str, Any]) -> None:
    if not isinstance(marker, dict):
        return
    mid = marker.get("id")
    if not mid:
        raise ValueError("marker missing id")

    attrs = {
        "id": str(mid),
        "markerWidth": _fmt_num(marker.get("markerWidth", 10)),
        "markerHeight": _fmt_num(marker.get("markerHeight", 10)),
        "refX": _fmt_num(marker.get("refX", 0)),
        "refY": _fmt_num(marker.get("refY", 0)),
        "orient": str(marker.get("orient", "auto")),
    }
    if marker.get("markerUnits"):
        attrs["markerUnits"] = str(marker["markerUnits"])

    mel = SubElement(parent, "marker", attrs)
    # viewBox optional for marker
    if marker.get("viewBox"):
        mel.set("viewBox", str(marker["viewBox"]))

    path_d = marker.get("pathD")
    if not path_d:
        raise ValueError(f"marker {mid} missing pathD")

    pel = SubElement(mel, "path", {"d": str(path_d)})
    _apply_style(pel, marker.get("style") or {})


def _emit_gradient(parent: Element, grad: Dict[str, Any]) -> None:
    if not isinstance(grad, dict):
        return
    gid = grad.get("id")
    if not gid:
        raise ValueError("gradient missing id")

    gtype = grad.get("type", "linear")
    if gtype not in {"linear", "radial"}:
        raise ValueError(f"unsupported gradient type: {gtype}")

    tag = "linearGradient" if gtype == "linear" else "radialGradient"
    gel = SubElement(parent, tag, {"id": str(gid)})

    # units + coords
    if grad.get("gradientUnits"):
        gel.set("gradientUnits", str(grad["gradientUnits"]))

    for k in ("x1", "y1", "x2", "y2", "cx", "cy", "r", "fx", "fy"):
        if k in grad and grad[k] is not None:
            gel.set(k, _fmt_num(grad[k]))

    # stops
    stops = grad.get("stops") or []
    if not isinstance(stops, list) or not stops:
        raise ValueError(f"gradient {gid} missing stops")

    for s in stops:
        if not isinstance(s, dict):
            continue
        off = s.get("offset")
        col = s.get("color")
        if off is None or col is None:
            continue
        sel = SubElement(gel, "stop", {"offset": _fmt_offset(off), "stop-color": str(col)})
        if s.get("opacity") is not None:
            sel.set("stop-opacity", _fmt_num(s["opacity"]))


def _emit_clip_path(parent: Element, cp: Dict[str, Any]) -> None:
    if not isinstance(cp, dict):
        return
    cid = cp.get("id")
    if not cid:
        raise ValueError("clipPath missing id")
    cel = SubElement(parent, "clipPath", {"id": str(cid)})
    for node in cp.get("elements") or []:
        _emit_node(cel, node)


def _emit_node(parent: Element, node: Dict[str, Any]) -> None:
    if not isinstance(node, dict):
        raise TypeError("element node must be an object")

    etype = node.get("type")
    if not etype:
        raise ValueError("element missing type")

    if etype == "group":
        el = SubElement(parent, "g")
        _apply_common(el, node)
        for child in node.get("children") or []:
            _emit_node(el, child)
        return

    tag = _map_type_to_tag(etype)
    el = SubElement(parent, tag)
    _apply_common(el, node)

    if etype == "rect":
        _set(el, "x", node.get("x")); _set(el, "y", node.get("y"))
        _set(el, "width", node.get("width")); _set(el, "height", node.get("height"))
        if node.get("rx") is not None: _set(el, "rx", node.get("rx"))
        if node.get("ry") is not None: _set(el, "ry", node.get("ry"))

    elif etype == "circle":
        _set(el, "cx", node.get("cx")); _set(el, "cy", node.get("cy")); _set(el, "r", node.get("r"))

    elif etype == "ellipse":
        _set(el, "cx", node.get("cx")); _set(el, "cy", node.get("cy")); _set(el, "rx", node.get("rx")); _set(el, "ry", node.get("ry"))

    elif etype == "line":
        _set(el, "x1", node.get("x1")); _set(el, "y1", node.get("y1")); _set(el, "x2", node.get("x2")); _set(el, "y2", node.get("y2"))

    elif etype in {"polyline", "polygon"}:
        pts = node.get("points")
        if not isinstance(pts, list) or not pts:
            raise ValueError(f"{etype} missing points")
        el.set("points", _fmt_points(pts))

    elif etype == "path":
        d = node.get("d")
        if not d:
            raise ValueError("path missing d")
        el.set("d", str(d))

    elif etype == "text":
        _emit_text(el, node)

    elif etype == "image":
        _set(el, "x", node.get("x")); _set(el, "y", node.get("y"))
        _set(el, "width", node.get("width")); _set(el, "height", node.get("height"))
        href = node.get("href")
        if not href:
            raise ValueError("image missing href")
        el.set("href", str(href))
        # Older viewers
        el.set(f"{{{_XLINK_NS}}}href", str(href))

    else:
        raise ValueError(f"unsupported element type: {etype}")


def _emit_text(el: Element, node: Dict[str, Any]) -> None:
    _set(el, "x", node.get("x")); _set(el, "y", node.get("y"))

    anchor = node.get("anchor")
    baseline = node.get("baseline")
    if anchor:
        el.set("text-anchor", str(anchor))
    if baseline:
        el.set("dominant-baseline", str(baseline))

    # Allow explicit lines for deterministic layout
    lines = node.get("lines")
    if lines is None:
        text = node.get("text", "")
        if "\n" in str(text):
            lines = str(text).split("\n")
        else:
            lines = [str(text)]

    if not isinstance(lines, list):
        lines = [str(lines)]

    lh = float(node.get("lineHeight", 1.2))
    font_size = _style_font_size(node.get("style") or {})

    # First line as text, rest as tspans with dy
    if not lines:
        return
    el.text = str(lines[0])

    if len(lines) > 1:
        dy = (font_size * lh) if font_size else (16 * lh)
        for i, line in enumerate(lines[1:], start=1):
            tspan = SubElement(el, "tspan", {"x": el.get("x", "0"), "dy": _fmt_num(dy)})
            tspan.text = str(line)


def _apply_common(el: Element, node: Dict[str, Any]) -> None:
    if node.get("id"):
        el.set("id", str(node["id"]))

    # style
    _apply_style(el, node.get("style") or {})

    # transform
    tf = node.get("transform")
    if tf:
        el.set("transform", _fmt_transform(tf))

    # clip-path convenience
    cp = (node.get("clipPath") or node.get("clip-path"))
    if cp:
        el.set("clip-path", f"url(#{cp})" if not str(cp).startswith("url(") else str(cp))


_STYLE_KEY_MAP = {
    "strokeWidth": "stroke-width",
    "strokeLinecap": "stroke-linecap",
    "strokeLinejoin": "stroke-linejoin",
    "strokeDasharray": "stroke-dasharray",
    "strokeDashoffset": "stroke-dashoffset",
    "opacity": "opacity",
    "fillOpacity": "fill-opacity",
    "strokeOpacity": "stroke-opacity",
    "fontFamily": "font-family",
    "fontSize": "font-size",
    "fontWeight": "font-weight",
    "letterSpacing": "letter-spacing",
    "vectorEffect": "vector-effect",
    "shapeRendering": "shape-rendering",
    "textRendering": "text-rendering",
    "markerEnd": "marker-end",
    "markerStart": "marker-start",
    "markerMid": "marker-mid",
}


def _apply_style(el: Element, style: Dict[str, Any]) -> None:
    if not isinstance(style, dict):
        return

    # accept snake_case or kebab-case too
    for k, v in style.items():
        if v is None:
            continue

        kk = str(k)
        if kk in _STYLE_KEY_MAP:
            kk = _STYLE_KEY_MAP[kk]
        kk = kk.replace("_", "-")

        if kk in {"stroke-width", "font-size", "opacity", "fill-opacity", "stroke-opacity", "letter-spacing"}:
            el.set(kk, _fmt_num(v))
        else:
            el.set(kk, str(v))


def _fmt_transform(tf: Any) -> str:
    if isinstance(tf, str):
        return tf
    if not isinstance(tf, list):
        raise TypeError("transform must be a list of ops")

    parts: List[str] = []
    for op in tf:
        if not isinstance(op, dict) or len(op) != 1:
            raise ValueError("each transform op must be single-key object")
        name, val = next(iter(op.items()))
        name = str(name)
        if name == "translate":
            x, y = _pair(val)
            parts.append(f"translate({_fmt_num(x)} {_fmt_num(y)})")
        elif name == "scale":
            if isinstance(val, (int, float)):
                parts.append(f"scale({_fmt_num(val)})")
            else:
                x, y = _pair(val)
                parts.append(f"scale({_fmt_num(x)} {_fmt_num(y)})")
        elif name == "rotate":
            vals = list(val) if isinstance(val, (list, tuple)) else [val]
            if len(vals) == 1:
                parts.append(f"rotate({_fmt_num(vals[0])})")
            elif len(vals) == 3:
                parts.append(f"rotate({_fmt_num(vals[0])} {_fmt_num(vals[1])} {_fmt_num(vals[2])})")
            else:
                raise ValueError("rotate expects [angle] or [angle,cx,cy]")
        elif name == "skewX":
            parts.append(f"skewX({_fmt_num(val)})")
        elif name == "skewY":
            parts.append(f"skewY({_fmt_num(val)})")
        elif name == "matrix":
            vals = list(val)
            if len(vals) != 6:
                raise ValueError("matrix expects 6 numbers")
            parts.append("matrix(" + " ".join(_fmt_num(x) for x in vals) + ")")
        else:
            raise ValueError(f"unsupported transform op: {name}")

    return " ".join(parts)


def _map_type_to_tag(etype: str) -> str:
    mapping = {
        "rect": "rect",
        "circle": "circle",
        "ellipse": "ellipse",
        "line": "line",
        "polyline": "polyline",
        "polygon": "polygon",
        "path": "path",
        "text": "text",
        "image": "image",
    }
    if etype not in mapping:
        raise ValueError(f"unsupported element type: {etype}")
    return mapping[etype]


def _set(el: Element, attr: str, val: Any) -> None:
    if val is None:
        raise ValueError(f"missing required attribute: {attr}")
    el.set(attr, _fmt_num(val))


def _fmt_points(points: List[Any]) -> str:
    out = []
    for p in points:
        x, y = _pair(p)
        out.append(f"{_fmt_num(x)},{_fmt_num(y)}")
    return " ".join(out)


def _pair(val: Any) -> Tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    raise ValueError("expected [x,y]")


def _fmt_num(v: Any) -> str:
    try:
        f = float(v)
    except Exception as e:
        raise ValueError(f"expected number, got {v!r}") from e
    if not math.isfinite(f):
        raise ValueError("non-finite number")
    # limit precision to keep SVG size sane
    s = f"{f:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_len(v: Any, units: str) -> str:
    # Allow percent strings etc.
    if isinstance(v, str):
        return v
    return _fmt_num(v) + str(units or "")


def _fmt_offset(off: Any) -> str:
    if isinstance(off, str):
        return off
    f = float(off)
    if 0 <= f <= 1:
        return f"{f * 100:.2f}%".rstrip("0").rstrip(".") + "%"
    return _fmt_num(f)


def _assert_viewbox(vb: Any) -> None:
    if isinstance(vb, str):
        parts = vb.split()
    elif isinstance(vb, (list, tuple)):
        parts = list(vb)
        vb = " ".join(str(x) for x in parts)
    else:
        raise ValueError("viewBox must be a string or [minX,minY,w,h]")

    if len(parts) != 4:
        raise ValueError("viewBox must have 4 numbers")
    nums = [float(x) for x in parts]
    if nums[2] <= 0 or nums[3] <= 0:
        raise ValueError("viewBox width/height must be > 0")


def _style_font_size(style: Dict[str, Any]) -> Optional[float]:
    if not isinstance(style, dict):
        return None
    fs = style.get("fontSize") or style.get("font-size") or style.get("font_size")
    try:
        return float(fs) if fs is not None else None
    except Exception:
        return None
