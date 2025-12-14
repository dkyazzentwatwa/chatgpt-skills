#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

from svg_skill import build_svg
from svg_skill.validate import validate_svg
from svg_skill.render import render_png


def main() -> int:
    base = os.path.join(os.path.dirname(__file__), "examples")
    out = os.path.join(os.path.dirname(__file__), "_out")
    os.makedirs(out, exist_ok=True)

    specs = sorted(glob.glob(os.path.join(base, "*.json")))
    if not specs:
        print("no example specs found")
        return 1

    ok = True
    for sp in specs:
        name = os.path.splitext(os.path.basename(sp))[0]
        spec = json.load(open(sp, "r", encoding="utf-8"))
        svg_text = build_svg(spec)
        svg_path = os.path.join(out, name + ".svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        rep = validate_svg(svg_path, from_file=True)
        if rep.get("errors"):
            ok = False
            print(f"FAIL {name}: {rep['errors']}")
        else:
            print(f"OK   {name}")

        # Optional render
        png_path = os.path.join(out, name + ".png")
        r = render_png(svg_path, png_path, scale=2.0, from_file=True)
        if r.get("ok"):
            print(f"     rendered {os.path.basename(png_path)}")
        else:
            print("     (render skipped)")

    print("done")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
