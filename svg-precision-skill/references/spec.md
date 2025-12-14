# SVG Spec (JSON)

This skill's scripts build SVGs from a strict JSON spec.

## Top-level

```json
{
  "canvas": {"width": 800, "height": 450, "viewBox": "0 0 800 450", "units": "px", "background": "#ffffff"},
  "defs": {"markers": [...], "gradients": [...], "clipPaths": [...]},
  "elements": [...],
  "metadata": {"title": "...", "desc": "..."}
}
```

- `canvas.viewBox` is **required** ("minX minY width height").
- `defs` is optional. Use ids to reference `url(#id)`.
- `elements` is required (list of scene nodes).

## Common fields

Every element supports:

- `id` (optional)
- `style` (optional): `{"fill": "#...", "stroke": "#...", "strokeWidth": 2, "opacity": 1, "strokeLinecap": "round", "strokeLinejoin": "round", "fontFamily": "...", "fontSize": 14, "fontWeight": 400}`
- `transform` (optional): list of transforms, e.g. `[{"translate": [10, 20]}, {"rotate": [45, 100, 100]}]`

## Element types

### Group

```json
{"type": "group", "id": "layer1", "style": {...}, "children": [ ... ]}
```

### Rect

```json
{"type": "rect", "x": 10, "y": 10, "width": 100, "height": 40, "rx": 8, "ry": 8, "style": {...}}
```

### Circle / Ellipse

```json
{"type": "circle", "cx": 50, "cy": 50, "r": 20, "style": {...}}
{"type": "ellipse", "cx": 50, "cy": 50, "rx": 30, "ry": 20, "style": {...}}
```

### Line / Polyline / Polygon

```json
{"type": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 100, "style": {...}}
{"type": "polyline", "points": [[0,0],[10,10],[20,0]], "style": {...}}
{"type": "polygon", "points": [[0,0],[10,10],[20,0]], "style": {...}}
```

### Path

```json
{"type": "path", "d": "M10 10 L50 10 L50 50 Z", "style": {...}}
```

### Text

```json
{"type": "text", "x": 100, "y": 100, "text": "Hello", "style": {"fontSize": 16, "fill": "#111"},
 "anchor": "start", "baseline": "alphabetic", "maxWidth": 200, "lineHeight": 1.2}
```

Anchors: `start | middle | end`
Baselines: `alphabetic | middle | hanging | central`

### Image

```json
{"type": "image", "x": 0, "y": 0, "width": 200, "height": 100, "href": "data:image/png;base64,..."}
```

## Defs templates

### Arrow marker

```json
{
  "defs": {
    "markers": [
      {"id": "arrow", "markerWidth": 10, "markerHeight": 10, "refX": 9, "refY": 5,
       "orient": "auto", "pathD": "M0 0 L10 5 L0 10 Z", "style": {"fill": "#444"}}
    ]
  }
}
```

Use on a line/path: `"style": {"markerEnd": "url(#arrow)"}`

### Linear gradient

```json
{
  "defs": {
    "gradients": [
      {"type": "linear", "id": "grad1", "x1": 0, "y1": 0, "x2": 1, "y2": 1,
       "stops": [{"offset": 0, "color": "#fff"}, {"offset": 1, "color": "#ddd"}]}
    ]
  }
}
```

Use: `"fill": "url(#grad1)"`

---

# Ready-to-copy templates

## Icon (24x24 stroke)

```json
{
  "canvas": {"width": 24, "height": 24, "viewBox": "0 0 24 24", "units": "px"},
  "elements": [
    {"type": "path", "d": "M4 12 L20 12", "style": {"fill": "none", "stroke": "#111", "strokeWidth": 2, "strokeLinecap": "round"}}
  ]
}
```

## Diagram (nodes + arrows)

```json
{
  "canvas": {"width": 1200, "height": 800, "viewBox": "0 0 1200 800", "units": "px", "background": "#fff"},
  "defs": {"markers": [{"id": "arrow", "markerWidth": 10, "markerHeight": 10, "refX": 9, "refY": 5, "orient": "auto",
    "pathD": "M0 0 L10 5 L0 10 Z", "style": {"fill": "#444"}}]},
  "elements": [
    {"type": "rect", "x": 120, "y": 120, "width": 220, "height": 80, "rx": 12, "ry": 12,
     "style": {"fill": "#f7f7f7", "stroke": "#444", "strokeWidth": 2}},
    {"type": "text", "x": 230, "y": 168, "text": "Start", "anchor": "middle", "baseline": "middle",
     "style": {"fill": "#111", "fontSize": 20, "fontFamily": "Inter, system-ui, sans-serif"}},
    {"type": "line", "x1": 340, "y1": 160, "x2": 520, "y2": 160,
     "style": {"stroke": "#444", "strokeWidth": 2, "markerEnd": "url(#arrow)"}}
  ]
}
```

## Chart (bar)

```json
{
  "canvas": {"width": 800, "height": 450, "viewBox": "0 0 800 450", "units": "px", "background": "#fff"},
  "elements": [
    {"type": "rect", "x": 100, "y": 50, "width": 600, "height": 320, "style": {"fill": "none", "stroke": "#ddd"}},
    {"type": "rect", "x": 140, "y": 220, "width": 80, "height": 150, "style": {"fill": "#4c6ef5"}},
    {"type": "rect", "x": 260, "y": 180, "width": 80, "height": 190, "style": {"fill": "#4c6ef5"}},
    {"type": "text", "x": 400, "y": 410, "text": "Q1 - Q2 - Q3", "anchor": "middle", "baseline": "middle", "style": {"fill": "#666", "fontSize": 14}}
  ]
}
```

## UI mockup (card)

```json
{
  "canvas": {"width": 1440, "height": 900, "viewBox": "0 0 1440 900", "units": "px", "background": "#f2f2f2"},
  "elements": [
    {"type": "rect", "x": 360, "y": 180, "width": 720, "height": 420, "rx": 24, "ry": 24,
     "style": {"fill": "#ffffff", "stroke": "#e6e6e6"}},
    {"type": "text", "x": 420, "y": 260, "text": "Settings", "style": {"fill": "#111", "fontSize": 28, "fontWeight": 700}}
  ]
}
```

## Technical drawing (dimension line)

```json
{
  "canvas": {"width": 900, "height": 600, "viewBox": "0 0 900 600", "units": "mm", "background": "#fff"},
  "defs": {"markers": [{"id": "dimArrow", "markerWidth": 8, "markerHeight": 8, "refX": 4, "refY": 4, "orient": "auto",
    "pathD": "M0 4 L4 0 L8 4 L4 8 Z", "style": {"fill": "#111"}}]},
  "elements": [
    {"type": "rect", "x": 200, "y": 200, "width": 300, "height": 140, "style": {"fill": "none", "stroke": "#111", "strokeWidth": 1}},
    {"type": "line", "x1": 200, "y1": 370, "x2": 500, "y2": 370,
     "style": {"stroke": "#111", "strokeWidth": 1, "markerStart": "url(#dimArrow)", "markerEnd": "url(#dimArrow)"}},
    {"type": "text", "x": 350, "y": 395, "text": "300 mm", "anchor": "middle", "style": {"fill": "#111", "fontSize": 16}}
  ]
}
```
