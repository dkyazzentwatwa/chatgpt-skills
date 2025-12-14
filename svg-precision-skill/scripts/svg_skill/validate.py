from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from xml.etree import ElementTree as ET

_URL_REF_RE = re.compile(r"url\(#(?P<id>[A-Za-z_][\w:.-]*)\)")


def validate_svg(svg: Union[str, bytes], *, from_file: bool = False) -> Dict[str, Any]:
    """Validate an SVG string or a file path.

    Returns: {"errors": [...], "warnings": [...]}.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if from_file:
        path = str(svg)
        try:
            data = open(path, "rb").read()
        except Exception as e:
            return {"errors": [f"cannot read SVG file: {e}"], "warnings": []}
    else:
        data = svg if isinstance(svg, (bytes, bytearray)) else str(svg).encode("utf-8")

    try:
        root = ET.fromstring(data)
    except Exception as e:
        return {"errors": [f"invalid XML: {e}"], "warnings": []}

    tag = _strip_ns(root.tag)
    if tag != "svg":
        errors.append(f"root element is <{tag}>, expected <svg>")

    view_box = root.attrib.get("viewBox")
    if not view_box:
        errors.append("missing viewBox")
    else:
        vb_ok, vb_msg = _check_viewbox(view_box)
        if not vb_ok:
            errors.append(vb_msg)

    # width/height not strictly required but a quality signal
    if not root.attrib.get("width") or not root.attrib.get("height"):
        warnings.append("missing width/height on root <svg> (may render with unexpected scaling)")

    # collect ids
    ids: Set[str] = set()
    dup_ids: Set[str] = set()
    for el in root.iter():
        eid = el.attrib.get("id")
        if eid:
            if eid in ids:
                dup_ids.add(eid)
            ids.add(eid)

    if dup_ids:
        errors.append(f"duplicate ids: {', '.join(sorted(dup_ids))}")

    # find url(#id) references in attributes
    refs: Set[str] = set()
    for el in root.iter():
        for _, v in el.attrib.items():
            for m in _URL_REF_RE.finditer(v):
                refs.add(m.group("id"))

    missing = sorted(r for r in refs if r not in ids)
    if missing:
        errors.append("missing referenced ids: " + ", ".join(missing))

    # text gotchas
    for el in root.iter():
        if _strip_ns(el.tag) == "text":
            if "font-family" not in el.attrib and "style" not in el.attrib:
                warnings.append("text element without explicit font-family may render differently across systems")
                break

    return {"errors": errors, "warnings": warnings}


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _check_viewbox(vb: str) -> Tuple[bool, str]:
    parts = str(vb).split()
    if len(parts) != 4:
        return False, "viewBox must have 4 numbers"
    try:
        nums = [float(x) for x in parts]
    except Exception:
        return False, "viewBox contains non-numeric values"
    if nums[2] <= 0 or nums[3] <= 0:
        return False, "viewBox width/height must be > 0"
    return True, "ok"
