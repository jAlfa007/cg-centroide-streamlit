from __future__ import annotations
from typing import Dict, List, Tuple

def _svg_header(w: float, h: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="100%" '
        f'style="height:auto;display:block" preserveAspectRatio="xMidYMid meet" role="img">'
    )

def _svg_footer() -> str:
    return "</svg>"

def _style(fill_color: str, stroke_color: str, axis_color: str, grid_color: str = "#0a66c2", grid_top_color: str = "#ffffff") -> str:
    return f"""
  <style>
    .solid-bg {{ fill: #fff; stroke: none; }}
    .axis  {{ stroke: {axis_color}; stroke-width: 1.5; marker-end: url(#arrow-blue); }}
    .grid  {{ stroke: {grid_color}; stroke-opacity: 0.15; stroke-width: 1; }}
    .grid-top {{ stroke: {grid_top_color}; stroke-opacity: 0.8; stroke-width: 0.5; stroke-dasharray: 2,2; }}
    .contour-fill {{ fill: {fill_color}; stroke: {stroke_color}; stroke-width: 2; fill-rule: evenodd; }}
    .txt   {{ font-family: Arial, sans-serif; font-size: 12px; fill: #000; }}
    .txt-blue {{ font-family: Arial, sans-serif; font-size: 12px; fill: {axis_color}; }}
    .txt-label {{ font-family: Arial, sans-serif; font-size: 12px; fill: #111; font-weight: 600; }}
  </style>
  <defs>
    <marker id="arrow-blue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="{axis_color}" />
    </marker>
  </defs>
"""

def _line(x1,y1,x2,y2, klass="axis"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{klass}" />'

def _text(x,y, s, anchor="middle", klass="txt"):
    return f'<text x="{x}" y="{y}" class="{klass}" text-anchor="{anchor}">{s}</text>'

def _rect(x,y,w,h, klass="solid-bg"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{klass}" />'

def _path_from_polyline(points) -> str:
    if not points:
        return ""
    d = [f"M {points[0][0]} {points[0][1]}"]
    for (x,y) in points[1:]:
        d.append(f"L {x} {y}")
    d.append("Z")
    return " ".join(d)

def _path_from_rings(rings: List[List[Tuple[float,float]]]) -> str:
    """Convierte varias polilíneas (anillos) en un único 'd' de path.
       El primer anillo debe ser el contorno exterior; los siguientes, huecos."""
    parts: List[str] = []
    for pts in rings:
        if not pts:
            continue
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        parts.append(_path_from_polyline(pts))
    return " ".join(parts)

def exercise_to_svg(
    ex: Dict,
    margin: float = 30.0,
    show_centroid: bool = False,
    fill_color: str = "#222222",
    stroke_color: str = "#000000",
    axis_color: str = "#0a66c2",
    grid_color: str = "#0a66c2",
    show_grid: bool = False,
    show_grid_top: bool = False,
    grid_top_color: str = "#ffffff",
) -> str:
    grid = ex["grid"]
    cs   = grid["cell_size"]
    gw   = grid["width"]
    gh   = grid["height"]

    base_grid   = gw * cs
    height_grid = gh * cs

    width = base_grid + 2*margin
    hght  = height_grid + 2*margin

    svg = [_svg_header(width, hght), _style(fill_color, stroke_color, axis_color, grid_color, grid_top_color)]
    svg.append(_rect(0,0,width,hght,"solid-bg"))

    # Ejes
    origin_x = margin
    origin_y = margin + height_grid
    svg.append(_line(origin_x, origin_y, origin_x + base_grid + 10, origin_y, "axis"))
    svg.append(_text(origin_x + base_grid + 15, origin_y + 5, "X", anchor="start", klass="txt-blue"))
    svg.append(_line(origin_x, origin_y, origin_x, origin_y - (height_grid + 10), "axis"))
    svg.append(_text(origin_x - 7, origin_y - (height_grid + 15), "Y", anchor="end", klass="txt-blue"))

    # Grilla debajo
    if show_grid:
        for i in range(gw + 1):
            x = origin_x + i * cs
            svg.append(_line(x, origin_y, x, origin_y - height_grid, "grid"))
        for j in range(gh + 1):
            y = origin_y - j * cs
            svg.append(_line(origin_x, y, origin_x + base_grid, y, "grid"))

    # Contorno + huecos como un único PATH (fill-rule: evenodd)
    rings: List[List[Tuple[float,float]]] = []
    if "contour" in ex and ex["contour"].get("vertices"):
        outer = []
        for (x,y) in ex["contour"]["vertices"]:
            sx = origin_x + x
            sy = origin_y - y
            outer.append((sx, sy))
        rings.append(outer)

    # Huecos
    for hole in ex.get("holes", []):
        pts = []
        for (x,y) in hole:
            sx = origin_x + x
            sy = origin_y - y
            pts.append((sx, sy))
        if pts:
            rings.append(pts)

    if rings:
        d = _path_from_rings(rings)
        svg.append(f'<path d="{d}" class="contour-fill" />')

    # Grilla superior (solo sobre la figura sólida)
    if show_grid_top:
        # Solo dibujar líneas de grilla que intersecten con la figura
        # Para esto, usamos el path del contorno como máscara
        if rings:
            # Crear un clipPath para la figura
            clip_id = f"clip-{hash(str(rings)) % 10000}"
            svg.append(f'<defs><clipPath id="{clip_id}"><path d="{_path_from_rings(rings)}" /></clipPath></defs>')
            
            # Agrupar las líneas de grilla superior y aplicar el clipPath
            svg.append(f'<g clip-path="url(#{clip_id})">')
            
            # Líneas verticales
            for i in range(gw + 1):
                x = origin_x + i * cs
                svg.append(_line(x, origin_y, x, origin_y - height_grid, "grid-top"))
            
            # Líneas horizontales
            for j in range(gh + 1):
                y = origin_y - j * cs
                svg.append(_line(origin_x, y, origin_x + base_grid, y, "grid-top"))
            
            svg.append('</g>')

    # Centroide
    if show_centroid and "solution" in ex:
        cx = ex["solution"]["cx"]; cy = ex["solution"]["cy"]
        xg = origin_x + cx; yg = origin_y - cy
        svg.append(f'<circle cx="{xg}" cy="{yg}" r="3" fill="#ff0000" stroke="none" />')

    svg.append(_svg_footer())
    return "\n".join(svg)
