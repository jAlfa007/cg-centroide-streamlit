from __future__ import annotations
from typing import Dict, Tuple, List, Set, Deque, Optional, DefaultDict, Callable, Any
from collections import deque, defaultdict
import math

from .seed import RNG, int_to_base36
from .contour import ContourParams, build_orthogonal_simple_contour

GEN_VERSION_CONTOUR = "cg-contour-2.6.2"

# ───────── Utilidades geométricas / listas ─────────

def _polygon_area_centroid(poly: List[Tuple[float,float]]) -> Tuple[float, float, float]:
    if len(poly) < 3:
        return (0.0, 0.0, 0.0)
    A = 0.0; Cx = 0.0; Cy = 0.0
    for i in range(len(poly)):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1) % len(poly)]
        cross = x1*y2 - x2*y1
        A  += cross
        Cx += (x1 + x2) * cross
        Cy += (y1 + y2) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return (0.0, 0.0, 0.0)
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (A, Cx, Cy)

def _area_centroid_circle(cx: float, cy: float, r: float) -> Tuple[float, float, float]:
    A = math.pi * r * r
    return (A, cx, cy)  # centroide = centro

def _composite_area_centroid_exact(outer_pts, holes_meta):
    """
    Centroide compuesto exacto:
    - contorno exterior por fórmula poligonal
    - círculos: π r^2 en (cx, cy)
    - polígonos (triángulos/cuadrados): fórmula poligonal
    """
    def _ring(poly):
        if not poly:
            return (0.0, 0.0, 0.0)
        if poly[0] != poly[-1]:
            poly = poly + [poly[0]]
        return _polygon_area_centroid(poly)

    A0, Cx0, Cy0 = _ring(outer_pts)
    A_net = A0
    Mx = A0 * Cx0
    My = A0 * Cy0

    for h in (holes_meta or []):
        kind = h.get("type")
        if kind == "circle":
            Ah, Cxh, Cyh = _area_centroid_circle(h["cx"], h["cy"], h["r"])
        else:
            # 'polygon' (triángulo/cuadrado) u otros que vengan como puntos
            Ah, Cxh, Cyh = _ring(h["points"])
        # restar hueco
        A_net -= Ah
        Mx    -= Ah * Cxh
        My    -= Ah * Cyh

    if abs(A_net) < 1e-12:
        return (0.0, 0.0, 0.0)
    return (A_net, Mx / A_net, My / A_net)

def _translate_vertices(verts:List[Tuple[float,float]], dx:float, dy:float)->List[Tuple[float,float]]:
    return [(x+dx, y+dy) for (x,y) in verts]

def _snap_to_grid_xy(x: float, y: float, cs: float) -> Tuple[float,float]:
    sx = round(x / cs) * cs
    sy = round(y / cs) * cs
    return (sx, sy)

def _append_snapped(pts: List[Tuple[float,float]], p: Tuple[float,float], cs: float):
    sp = _snap_to_grid_xy(p[0], p[1], cs)
    if not pts:
        pts.append(sp); return
    lx, ly = pts[-1]
    if abs(lx - sp[0]) > 1e-9 or abs(ly - sp[1]) > 1e-9:
        pts.append(sp)

def _append_raw(pts: List[Tuple[float,float]], p: Tuple[float,float]):
    if not pts:
        pts.append(p); return
    lx, ly = pts[-1]
    if abs(lx - p[0]) > 1e-9 or abs(ly - p[1]) > 1e-9:
        pts.append(p)

def _dedupe_colinear(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if len(pts) <= 3:
        return pts[:]
    out = [pts[0]]
    for i in range(1, len(pts)-1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i+1]
        if abs(bx-ax) < 1e-9 and abs(by-ay) < 1e-9:
            continue
        if abs((bx-ax)*(cy-by) - (by-ay)*(cx-bx)) < 1e-9:
            continue
        out.append((bx,by))
    if abs(out[-1][0]-pts[-1][0]) > 1e-9 or abs(out[-1][1]-pts[-1][1]) > 1e-9:
        out.append(pts[-1])
    return out

# ───────── Point-in-polygon ─────────

def _point_in_polygon(x: float, y: float, poly: List[Tuple[float,float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if xin > x:
                inside = not inside
    return inside

# ───────── Huecos internos (helpers) ─────────

def _generate_hole(kind: str, cx: float, cy: float, size_units: int, rng, cs: float) -> List[Tuple[float, float]]:
    """
    Genera un hueco con restricciones 'didácticas':
    - triangle: triángulo rectángulo ortogonal, vértices en nodos (múltiplos de cs).
               El ángulo recto está en (cx, cy); catetos de 'size_units' celdas.
    - quad: cuadrado ortogonal, vértices en nodos; lado = size_units * cs.
    - circle: círculo aproximado con 'steps' puntos, centro en nodo y radio = size_units * cs.
              (El polígono de aproximación no cae en nodos, pero el centro y el radio sí).
    """
    pts: List[Tuple[float, float]] = []

    # Aseguramos centro en nodo de grilla
    cx = round(cx / cs) * cs
    cy = round(cy / cs) * cs
    L = max(1, int(size_units)) * cs

    if kind == "triangle":
        # Elige uno de los 4 cuadrantes para los catetos
        # 0: +x,+y ; 1: -x,+y ; 2: -x,-y ; 3: +x,-y
        q = rng.randint(0, 3)
        if q == 0:
            a = (cx,     cy)      # vértice recto
            b = (cx+L,   cy)      # sobre +x
            c = (cx,     cy+L)    # sobre +y
        elif q == 1:
            a = (cx,     cy)
            b = (cx-L,   cy)
            c = (cx,     cy+L)
        elif q == 2:
            a = (cx,     cy)
            b = (cx-L,   cy)
            c = (cx,     cy-L)
        else:
            a = (cx,     cy)
            b = (cx+L,   cy)
            c = (cx,     cy-L)
        pts = [a, b, c, a]

    elif kind == "quad":
        # Cuadrado ortogonal centrado en (cx, cy) pero con esquinas en nodos.
        # Desplazamos el "ancla" a la esquina inferior-izquierda para mantener vértices en nodos.
        # Lado = L, vértices en la grilla:
        half = L / 2.0
        # Para garantizar nodos exactos, forzamos que la esquina inferior-izquierda quede en nodo:
        x0 = cx - round(half / cs) * cs
        y0 = cy - round(half / cs) * cs
        x1 = x0 + L
        y1 = y0 + L
        # Asegurar snaps exactos
        x0 = round(x0 / cs) * cs; y0 = round(y0 / cs) * cs
        x1 = round(x1 / cs) * cs; y1 = round(y1 / cs) * cs
        pts = [(x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0)]

    elif kind == "circle":
        r = L
        # Steps adaptativos: más puntos para círculos más grandes
        # Mínimo 24, pero adaptativo al radio y tamaño de celda
        steps = max(24, int(2 * math.pi * r / max(cs/2, 1e-6)))
        for i in range(steps):
            th = 2.0 * math.pi * i / steps
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        pts.append(pts[0])

    return pts

def _generate_holes(params, rng, contour: List[Tuple[float, float]]) -> Tuple[
    List[List[Tuple[float, float]]],  # holes_pts
    List[Dict[str, Any]]              # holes_meta
]:
    """
    Coloca huecos cumpliendo:
    - No tocar ni cruzar segmentos del contorno (margen interno y chequeo de intersección).
    - No coincidir con nodos del contorno.
    - No solaparse con otros huecos (chequeo por círculos envolventes).
    - TRIÁNGULO: vértices en nodos y lados ortogonales.
    - CUADRADO: vértices en nodos y lados ortogonales.
    - CÍRCULO: centro en nodo y radio = k * cs (k entero). (Se dibuja poligonal, pero el centroide usa πr² exacto.)
    Retorna:
      - holes_pts: lista de polilíneas (cada hueco), para dibujo/render.
      - holes_meta: metadatos para cálculo exacto del centroide compuesto.
    """
    cs = float(getattr(params, "cell_size", 12.0))

    # Cantidades
    if getattr(params, "random_holes", False):
        n_cir = rng.randint(0, 3)
        n_tri = rng.randint(0, 2)
        n_quad = rng.randint(0, 2)
    else:
        n_cir = int(getattr(params, "num_circles", 0) or 0)
        n_tri = int(getattr(params, "num_triangles", 0) or 0)
        n_quad = int(getattr(params, "num_quads", 0) or 0)

    total = n_cir + n_tri + n_quad
    if total <= 0:
        return [], []

    # Bounding box del contorno (acota la búsqueda)
    xs = [x for (x, _) in contour]
    ys = [y for (_, y) in contour]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Margen interno para que no "toque" el borde de la figura
    margin = 2.0 * cs

    # Nodos del contorno para evitar coincidencias exactas
    cont_nodes = set((round(x / cs), round(y / cs)) for (x, y) in contour)

    # Segmentos del contorno para chequear contacto/intersección con huecos
    def _segments_of(poly: List[Tuple[float,float]]) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
        if not poly:
            return []
        pts = poly[:]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        return [(pts[i], pts[i+1]) for i in range(len(pts)-1)]

    contour_segs = _segments_of(contour)

    def _no_touch_contour(hpts: List[Tuple[float,float]]) -> bool:
        """True si el hueco NO toca ni cruza el contorno exterior."""
        if not hpts:
            return True
        hp = hpts[:]
        if hp[0] != hp[-1]:
            hp.append(hp[0])
        for i in range(len(hp)-1):
            p1, p2 = hp[i], hp[i+1]
            for (q1, q2) in contour_segs:
                if _seg_intersect(p1, p2, q1, q2):  # usa helper global del módulo
                    return False
        return True

    # Huecos ya colocados (para evitar solapamiento básico)
    placed_meta: List[Tuple[float, float, float]] = []  # (cx, cy, r_envolvente)

    def _valid_polygon(hpts: List[Tuple[float, float]]) -> bool:
        """Validación robusta de huecos:
        - Todos los vértices dentro del contorno con margen ε
        - Respeta margen contra bounding box
        - Verifica que no toque el contorno
        """
        if not hpts:
            return False
            
        # Margen interno para evitar tocar el contorno
        ε = 1e-9
        
        # Todos los puntos dentro del contorno y respetando margen contra bounding
        for (x, y) in hpts:
            if not _point_in_polygon(x, y, contour):
                return False
            if x <= xmin + margin + ε or x >= xmax - margin - ε or y <= ymin + margin + ε or y >= ymax - margin - ε:
                return False
        
        # Verificación adicional: que no toque el contorno
        return _no_touch_contour(hpts)

    def _no_node_coincidence(hpts: List[Tuple[float, float]]) -> bool:
        # Si algún vértice cae EXACTO en un nodo del contorno → descartar
        for (x, y) in hpts:
            if (round(x / cs), round(y / cs)) in cont_nodes:
                return False
        return True

    def _no_overlap(cx: float, cy: float, r_env: float) -> bool:
        # Evitar solape entre huecos usando círculos envolventes
        for (px, py, pr) in placed_meta:
            dx, dy = cx - px, cy - py
            if (dx * dx + dy * dy) < (r_env + pr + 0.25 * cs) ** 2:
                return False
        return True

    holes_pts: List[List[Tuple[float, float]]]= []
    holes_meta: List[Dict[str, Any]] = []

    # Helper para colocar k huecos de un tipo
    def _place(kind: str, count: int):
        nonlocal holes_pts, holes_meta, placed_meta
        tries_per_item = 80

        for _ in range(count):
            placed = False
            for _try in range(tries_per_item):
                # Centro candidato: en NODO de grilla dentro del bounding con margen
                gxmin = int(math.ceil((xmin + margin) / cs))
                gxmax = int(math.floor((xmax - margin) / cs))
                gymin = int(math.ceil((ymin + margin) / cs))
                gymax = int(math.floor((ymax - margin) / cs))
                if gxmin >= gxmax or gymin >= gymax:
                    break  # sin espacio útil

                gx = rng.randint(gxmin, gxmax)
                gy = rng.randint(gymin, gymax)
                cx = gx * cs
                cy = gy * cs

                # Tamaño en unidades de celda (entero), acotado a lo que quepa
                max_units = max(1, min(gx - gxmin, gxmax - gx, gy - gymin, gymax - gy))
                if max_units <= 0:
                    continue
                size_units = rng.randint(1, max(1, min(max_units, 4)))  # 1..4 por defecto
                L = size_units * cs

                # Geometría del hueco para DIBUJO
                hpts = _generate_hole(kind, cx, cy, size_units, rng, cs)

                # Radio envolvente para test de solape (aprox por caja/diagonal)
                if kind == "circle":
                    r_env = L
                elif kind == "quad":
                    r_env = L * math.sqrt(2) / 2.0
                else:  # triangle
                    r_env = L * math.sqrt(2) / 2.0

                # Validaciones
                if not _valid_polygon(hpts):
                    continue
                if not _no_node_coincidence(hpts):
                    continue
                if not _no_overlap(cx, cy, r_env):
                    continue

                # Aprobado
                holes_pts.append(hpts)

                # Metadatos para centroide exacto:
                if kind == "circle":
                    holes_meta.append({"type": "circle", "cx": cx, "cy": cy, "r": L})
                else:
                    holes_meta.append({"type": "polygon", "points": hpts})

                placed_meta.append((cx, cy, r_env))
                placed = True
                break

            # si no se pudo colocar este hueco en tries_per_item, se omite (no forzamos)

    # Colocar por tipo en orden "más rígido" primero ayuda a encontrar lugar
    _place("triangle", n_tri)
    _place("quad",     n_quad)
    _place("circle",   n_cir)

    return holes_pts, holes_meta

# ───────── Raster + 4-conexión ─────────

def _cells_inside_polygon(verts: List[Tuple[float,float]], gw: int, gh: int, cs: float) -> Set[Tuple[int,int]]:
    inside: Set[Tuple[int,int]] = set()
    for j in range(gh):
        cy = (j + 0.5) * cs
        for i in range(gw):
            cx = (i + 0.5) * cs
            if _point_in_polygon(cx, cy, verts):
                inside.add((i, j))
    return inside

def _is_four_connected(cells: Set[Tuple[int,int]]) -> bool:
    if not cells:
        return False
    visited: Set[Tuple[int,int]] = set()
    start = next(iter(cells))
    q: Deque[Tuple[int,int]] = deque([start])
    visited.add(start)
    while q:
        i,j = q.popleft()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (i+di, j+dj)
            if nb in cells and nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited == cells

def _valid_four_connected(verts: List[Tuple[float,float]], gw:int, gh:int, cs:float) -> bool:
    cells = _cells_inside_polygon(verts, gw, gh, cs)
    return _is_four_connected(cells)

# ───────── Chequeos locales ─────────

def _is_orthogonal_turn(a,b,c) -> bool:
    ax,ay=a; bx,by=b; cx,cy=c
    dx1, dy1 = bx-ax, by-ay
    dx2, dy2 = cx-bx, cy-by
    return ((abs(dx1) > 1e-9 and abs(dy1) < 1e-9) and (abs(dx2) < 1e-9 and abs(dy2) > 1e-9)) or \
           ((abs(dx1) < 1e-9 and abs(dy1) > 1e-9) and (abs(dx2) > 1e-9 and abs(dy2) < 1e-9))

def _segment_len_cells(p, q, cs: float) -> float:
    return (abs(q[0]-p[0]) + abs(q[1]-p[1])) / cs

def _edge_length_units(p:Tuple[float,float], q:Tuple[float,float], cs:float) -> int:
    return int(round(abs(q[0]-p[0])/cs + abs(q[1]-p[1])/cs))

def _min_straights_ok(verts: List[Tuple[float,float]], cs: float, smin: int) -> bool:
    if smin <= 1:
        return True
    n = len(verts)
    if n < 4:
        return True
    for i in range(1, len(verts)-1):
        a = verts[i-1]; b = verts[i]; c = verts[i+1]
        if _is_orthogonal_turn(a,b,c):
            if _segment_len_cells(b, c, cs) + 1e-9 < float(smin):
                return False
    a = verts[-2]; b = verts[-1]; c = verts[1]
    if _is_orthogonal_turn(a,b,c):
        if _segment_len_cells(b, c, cs) + 1e-9 < float(smin):
            return False
    return True

def _count_corners(verts: List[Tuple[float,float]]) -> int:
    if len(verts) < 4:
        return 0
    cnt = 0
    for i in range(1, len(verts)-1):
        a = verts[i-1]; b = verts[i]; c = verts[i+1]
        ax,ay=a; bx,by=b; cx,cy=c
        dx1, dy1 = bx-ax, by-ay
        dx2, dy2 = cx-bx, cy-by
        if abs(dx1*dy2 - dy1*dx2) > 1e-9:
            cnt += 1
    return cnt

# ───────── Intersecciones + “unión por punto” ─────────

def _orient(ax,ay, bx,by, cx,cy) -> float:
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

def _on_seg(ax,ay, bx,by, px,py) -> bool:
    return min(ax,bx)-1e-9 <= px <= max(ax,bx)+1e-9 and min(ay,by)-1e-9 <= py <= max(ay,by)+1e-9

def _seg_intersect(p1, p2, q1, q2) -> bool:
    a1x,a1y = p1; a2x,a2y = p2; b1x,b1y = q1; b2x,b2y = q2
    o1 = _orient(a1x,a1y, a2x,a2y, b1x,b1y)
    o2 = _orient(a1x,a1y, a2x,a2y, b2x,b2y)
    o3 = _orient(b1x,b1y, b2x,b2y, a1x,a1y)
    o4 = _orient(b1x,b1y, b2x,b2y, a2x,a2y)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) < 1e-9 and _on_seg(a1x,a1y,a2x,a2y,b1x,b1y): return True
    if abs(o2) < 1e-9 and _on_seg(a1x,a1y,a2x,a2y,b2x,b2y): return True
    if abs(o3) < 1e-9 and _on_seg(b1x,b1y,b2x,b2y,a1x,a1y): return True
    if abs(o4) < 1e-9 and _on_seg(b1x,b1y,b2x,b2y,a2x,a2y): return True
    return False

def _has_self_intersections(pts: List[Tuple[float,float]]) -> bool:
    if len(pts) < 4:
        return False
    n = len(pts)
    for i in range(n-1):
        p1 = pts[i]; p2 = pts[i+1]
        for j in range(i+1, n-1):
            if j == i or j == i-1: continue
            if (i == 0 and j == n-2):
                continue
            q1 = pts[j]; q2 = pts[j+1]
            share = (abs(p1[0]-q1[0])<1e-9 and abs(p1[1]-q1[1])<1e-9) or \
                    (abs(p1[0]-q2[0])<1e-9 and abs(p1[1]-q2[1])<1e-9) or \
                    (abs(p2[0]-q1[0])<1e-9 and abs(p2[1]-q1[1])<1e-9) or \
                    (abs(p2[0]-q2[0])<1e-9 and abs(p2[1]-q2[1])<1e-9)
            if share:
                if not (abs(i - j) == 1 or (i == 0 and j == n-2)):
                    return True
                continue
            if _seg_intersect(p1,p2,q1,q2):
                return True
    return False

def _has_vertex_retouch(pts: List[Tuple[float,float]]) -> bool:
    if len(pts) < 4:
        return False
    n = len(pts)
    seen: Dict[Tuple[float,float], int] = {}
    for i, p in enumerate(pts):
        key = (round(p[0], 6), round(p[1], 6))
        if key in seen:
            j = seen[key]
            if not (abs(i - j) == 1 or (i == n-1 and j == 0) or (i == 0 and j == n-1)):
                return True
        else:
            seen[key] = i
    return False

# ───────── Mezclas / hashes ─────────

def _mix64(x: int) -> int:
    x &= (1<<64)-1
    x ^= (x >> 30); x = (x * 0xBF58476D1CE4E5B9) & ((1<<64)-1)
    x ^= (x >> 27); x = (x * 0x94D049BB133111EB) & ((1<<64)-1)
    x ^= (x >> 31)
    return x & ((1<<64)-1)

def _fingerprint_exact_params(params: ContourParams, quarters_want:int, semis_want:int) -> int:
    z = 0xC0FFEE123456789
    vals = [
        params.nx, params.ny,
        int(round(params.cell_size*1000)),
        params.max_corners,
        1 if params.allow_diagonals_45 else 0,
        params.max_diagonals,
        params.min_straights_after_corner,
        params.min_diags_after_diag,
        1 if params.lock_diag_opposite else 0,
        quarters_want, semis_want,
    ]
    for v in vals:
        z = _mix64(z ^ (int(v) & ((1<<64)-1)))
    return z

def _derived_seed(base_seed_int: int, attempt: int) -> int:
    return _mix64(base_seed_int + 0x9E3779B97F4A7C15 * (attempt+1))

# ───────── Fallback determinista ─────────

def _emergency_non_rectangular(seed_int:int, params:ContourParams) -> List[Tuple[float,float]]:
    cs = params.cell_size
    gw = params.nx - 1
    gh = params.ny - 1
    smin = max(1, params.min_straights_after_corner)
    m = 2
    x0 = m * cs; y0 = m * cs
    x1 = (gw - m) * cs; y1 = (gh - m) * cs
    Wc = (gw - 2*m); Hc = (gh - 2*m)
    if Wc < 2*smin+1 or Hc < 2*smin+1:
        return [(0.0,0.0),(gw*cs,0.0),(gw*cs,gh*cs),(0.0,gh*cs),(0.0,0.0)]

    rng = RNG(seed_int ^ 0xA5A5A5A5)
    edge = rng.randint(0,3)
    if edge in (0,2):
        usable = max(1, Wc - 2*smin)
        inner = max(smin, min(usable, max(1, usable//2)))
        start = smin + (rng.randint(0, max(1, usable - inner)) if usable - inner > 0 else 0)
        depth = max(smin, max(1, Hc//4))
    else:
        usable = max(1, Hc - 2*smin)
        inner = max(smin, min(usable, max(1, usable//2)))
        start = smin + (rng.randint(0, max(1, usable - inner)) if usable - inner > 0 else 0)
        depth = max(smin, max(1, Wc//4))

    if edge == 0:
        sx = x0 + start*cs; ex = sx + inner*cs; d = depth*cs
        verts = [(x0,y0),(sx,y0),(sx,y0+d),(ex,y0+d),(ex,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]
    elif edge == 2:
        sx = x1 - start*cs; ex = sx - inner*cs; d = depth*cs
        verts = [(x0,y0),(x1,y0),(x1,y1),(sx,y1),(sx,y1-d),(ex,y1-d),(ex,y1),(x0,y1),(x0,y0)]
    elif edge == 1:
        sy = y0 + start*cs; ey = sy + inner*cs; d = depth*cs
        verts = [(x0,y0),(x1,y0),(x1,sy),(x1-d,sy),(x1-d,ey),(x1,ey),(x1,y1),(x0,y1),(x0,y0)]
    else:
        sy = y1 - start*cs; ey = sy - inner*cs; d = depth*cs
        verts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,sy),(x0+d,sy),(x0+d,ey),(x0,ey),(x0,y0)]
    return verts

# ───────── Utilidades de arcos ─────────

def _is_axis_aligned(p:Tuple[float,float], q:Tuple[float,float]) -> bool:
    return abs(p[0]-q[0]) < 1e-9 or abs(p[1]-q[1]) < 1e-9

def _unit_dir(p:Tuple[float,float], q:Tuple[float,float]) -> Tuple[int,int]:
    dx = q[0]-p[0]; dy = q[1]-p[1]
    if abs(dx) > abs(dy):
        return (1 if dx>0 else -1, 0)
    else:
        return (0, 1 if dy>0 else -1)

def _left_normal(d:Tuple[int,int]) -> Tuple[int,int]:
    return (-d[1], d[0])

def _right_normal(d:Tuple[int,int]) -> Tuple[int,int]:
    return (d[1], -d[0])

def _angle(cx,cy, x,y) -> float:
    return math.atan2(y-cy, x-cx)

def _normalize_to_pi(dth: float) -> float:
    while dth <= -math.pi:
        dth += 2*math.pi
    while dth > math.pi:
        dth -= 2*math.pi
    return dth

def _sample_arc_dtheta(cx: float, cy: float, r: float, th0: float, dth: float, max_step: float) -> List[Tuple[float,float]]:
    L = abs(dth) * r
    steps = max(8, int(math.ceil(L / max_step)))
    out = []
    for k in range(1, steps):
        t = k / steps
        th = th0 + dth * t
        out.append((cx + r*math.cos(th), cy + r*math.sin(th)))
    return out

# ───────── Candidatos de arcos ─────────

class SemiCandidate:
    def __init__(self, edge_index:int, start_u:int, end_u:int,
                 a:Tuple[float,float], b:Tuple[float,float],
                 center:Tuple[float,float], r_px:float, r_units:int, ccw:bool):
        self.edge_index=edge_index
        self.start_u=start_u
        self.end_u=end_u
        self.a=a; self.b=b; self.center=center
        self.r_px=r_px; self.r_units=r_units
        self.ccw=ccw

class QuarterCandidate:
    def __init__(self, corner_index:int, p1:Tuple[float,float], p2:Tuple[float,float],
                 center:Tuple[float,float], r_px:float, r_units:int, cw:bool):
        self.corner_index=corner_index
        self.p1=p1; self.p2=p2; self.center=center
        self.r_px=r_px; self.r_units=r_units
        self.cw=cw

def _find_candidates_quarters(verts:List[Tuple[float,float]], cs:float, R:int, smin:int) -> List[QuarterCandidate]:
    n = len(verts)
    out: List[QuarterCandidate] = []
    Rpx = R*cs
    for i in range(1, n-1):
        a = verts[i-1]; b = verts[i]; c = verts[i+1]
        if not (_is_axis_aligned(a,b) and _is_axis_aligned(b,c)):
            continue
        d1 = _unit_dir(a,b)
        d2 = _unit_dir(b,c)
        if abs(d1[0])==abs(d2[0]) or abs(d1[1])==abs(d2[1]):
            continue
        L1 = _edge_length_units(a,b,cs)
        L2 = _edge_length_units(b,c,cs)
        if L1 < R or L2 < R:
            continue

        cross = d1[0]*d2[1] - d1[1]*d2[0]
        p1 = _snap_to_grid_xy(b[0] - d1[0]*Rpx, b[1] - d1[1]*Rpx, cs)
        p2 = _snap_to_grid_xy(b[0] + d2[0]*Rpx, b[1] + d2[1]*Rpx, cs)

        if cross < 0:
            n1 = _left_normal(d1); n2 = _left_normal(d2)
            c1 = (p1[0] + n1[0]*Rpx, p1[1] + n1[1]*Rpx)
            c2 = (p2[0] + n2[0]*Rpx, p2[1] + n2[1]*Rpx)
            cx = 0.5*(c1[0]+c2[0]); cy = 0.5*(c1[1]+c2[1])
            out.append(QuarterCandidate(i, p1, p2, (cx,cy), Rpx, R, cw=True))
        else:
            n1 = _right_normal(d1); n2 = _right_normal(d2)
            c1 = (p1[0] + n1[0]*Rpx, p1[1] + n1[1]*Rpx)
            c2 = (p2[0] + n2[0]*Rpx, p2[1] + n2[1]*Rpx)
            cx = 0.5*(c1[0]+c2[0]); cy = 0.5*(c1[1]+c2[1])
            out.append(QuarterCandidate(i, p1, p2, (cx,cy), Rpx, R, cw=False))
    return out

def _find_candidates_semis(verts:List[Tuple[float,float]], cs:float, R:int, smin:int) -> List[SemiCandidate]:
    n = len(verts)
    out: List[SemiCandidate] = []
    Rpx = R*cs
    s_arc = max(1, smin - 1)
    for i in range(n-1):
        p = verts[i]; q = verts[i+1]
        if not _is_axis_aligned(p,q):
            continue
        L = _edge_length_units(p,q,cs)
        if L < 2*R + 2*s_arc:
            continue
        d = _unit_dir(p,q)
        nrm = _left_normal(d)
        t_min = s_arc
        t_max = L - 2*R - s_arc
        if t_min > t_max:
            continue
        for t in range(t_min, t_max+1):
            a = _snap_to_grid_xy(p[0] + d[0]*t*cs,          p[1] + d[1]*t*cs,          cs)
            b = _snap_to_grid_xy(p[0] + d[0]*(t+2*R)*cs,    p[1] + d[1]*(t+2*R)*cs,    cs)
            midx = (a[0] + b[0])/2.0; midy = (a[1] + b[1])/2.0
            center = (midx + nrm[0]*Rpx, midy + nrm[1]*Rpx)
            out.append(SemiCandidate(i, start_u=t, end_u=t+2*R,
                                     a=a, b=b, center=center, r_px=Rpx, r_units=R, ccw=True))
    return out

# ───────── Arcos: usar siempre arco menor ─────────

def _arc_points_quarter(c:QuarterCandidate, max_step:float) -> List[Tuple[float,float]]:
    cx,cy = c.center; r = c.r_px
    th1 = _angle(cx,cy, c.p1[0], c.p1[1])
    th2 = _angle(cx,cy, c.p2[0], c.p2[1])
    dth = _normalize_to_pi(th2 - th1)
    if abs(dth) > (3.0*math.pi/4.0):
        dth = math.copysign(math.pi/2.0, dth)
    if abs(abs(dth) - (math.pi/2.0)) > 1e-3:
        dth = math.copysign(math.pi/2.0, dth)
    return _sample_arc_dtheta(cx, cy, r, th1, dth, max_step)

def _arc_points_semi(c:SemiCandidate, max_step:float) -> List[Tuple[float,float]]:
    cx,cy = c.center; r = c.r_px
    th1 = _angle(cx,cy, c.a[0], c.a[1])
    th2 = _angle(cx,cy, c.b[0], c.b[1])
    dth = _normalize_to_pi(th2 - th1)
    if abs(abs(dth) - math.pi) > 1e-3:
        dth = math.pi if dth >= 0 else -math.pi
        if abs(dth) < 1e-6:
            dth = math.pi
    return _sample_arc_dtheta(cx, cy, r, th1, dth, max_step)

# ───────── Inserción de arcos ─────────

def _apply_arcs_multiR(
    verts: List[Tuple[float,float]],
    cs: float,
    seed_int: int,
    quarters_want: int,
    semis_want: int,
    smin: int,
) -> Tuple[List[Tuple[float,float]], Dict]:
    want_any_arc = (quarters_want > 0) or (semis_want > 0)
    if not want_any_arc:
        return verts, {"radius_units": {}, "quarters_placed": 0, "semis_placed": 0}

    gw_units = max(1, int(round(max(x for x,_ in verts)/cs)))
    gh_units = max(1, int(round(max(y for _,y in verts)/cs)))
    Rmax = max(1, min(6, gw_units//3, gh_units//3))

    rng = RNG(_derived_seed(seed_int, 97))

    remain_q = quarters_want
    remain_s = semis_want

    chosen_quarters: List[QuarterCandidate] = []
    chosen_semis: List[SemiCandidate] = []
    used_corners: Set[int] = set()
    used_edges_for_semis: Set[int] = set()

    radii_hist_q: DefaultDict[int,int] = defaultdict(int)
    radii_hist_s: DefaultDict[int,int] = defaultdict(int)

    n_edges = len(verts) - 1

    for R in range(Rmax, 0, -1):
        if remain_q <= 0 and remain_s <= 0:
            break

        q_cands = _find_candidates_quarters(verts, cs, R, smin)
        s_cands = _find_candidates_semis(verts, cs, R, smin)
        rng.shuffle(q_cands)
        rng.shuffle(s_cands)

        for q in q_cands:
            if remain_q <= 0: break
            if q.corner_index in used_corners: continue
            chosen_quarters.append(q)
            used_corners.add(q.corner_index)
            radii_hist_q[q.r_units] += 1
            remain_q -= 1

        blocked_edges: Set[int] = set()
        for q in chosen_quarters:
            left_e  = (q.corner_index-1) % n_edges
            right_e = (q.corner_index) % n_edges
            blocked_edges.add(left_e); blocked_edges.add(right_e)

        for s in s_cands:
            if remain_s <= 0: break
            ei = s.edge_index
            if ei in used_edges_for_semis or ei in blocked_edges:
                continue
            chosen_semis.append(s)
            used_edges_for_semis.add(ei)
            radii_hist_s[s.r_units] += 1
            remain_s -= 1

    if not chosen_quarters and not chosen_semis:
        for R in range(Rmax, 0, -1):
            q_c = _find_candidates_quarters(verts, cs, R, smin)
            s_c = _find_candidates_semis(verts, cs, R, smin)
            if q_c:
                q = q_c[0]; chosen_quarters.append(q); radii_hist_q[R]+=1; break
            if s_c:
                s = s_c[0]; chosen_semis.append(s); radii_hist_s[R]+=1; break

    max_step = cs * 0.35
    quarter_at = {qc.corner_index: qc for qc in chosen_quarters}
    semis_by_edge: Dict[int, List[SemiCandidate]] = {}
    for sc in chosen_semis:
        semis_by_edge.setdefault(sc.edge_index, []).append(sc)
    for ei in semis_by_edge:
        semis_by_edge[ei].sort(key=lambda s: s.start_u)

    new_poly: List[Tuple[float,float]] = []
    skip_next_start: Optional[Tuple[float,float]] = None

    n = len(verts)
    for i in range(n-1):
        a = verts[i]; b = verts[i+1]
        a_eff = skip_next_start if skip_next_start is not None else a
        skip_next_start = None
        _append_snapped(new_poly, a_eff, cs)

        semis_here = semis_by_edge.get(i, [])
        if semis_here and _is_axis_aligned(a,b):
            for sc in semis_here:
                _append_snapped(new_poly, sc.a, cs)
                for p in _arc_points_semi(sc, max_step=max_step):
                    _append_raw(new_poly, p)
                _append_snapped(new_poly, sc.b, cs)
            end_point = quarter_at[i+1].p1 if (i+1) in quarter_at else b
            _append_snapped(new_poly, end_point, cs)
        else:
            if (i+1) in quarter_at and _is_axis_aligned(a,b):
                _append_snapped(new_poly, quarter_at[i+1].p1, cs)
            else:
                _append_snapped(new_poly, b, cs)

        if (i+1) in quarter_at:
            qc = quarter_at[i+1]
            for p in _arc_points_quarter(qc, max_step=max_step):
                _append_raw(new_poly, p)
            _append_snapped(new_poly, qc.p2, cs)
            skip_next_start = qc.p2

    if abs(new_poly[0][0]-new_poly[-1][0]) > 1e-9 or abs(new_poly[0][1]-new_poly[-1][1]) > 1e-9:
        new_poly.append(new_poly[0])

    new_poly = _dedupe_colinear(new_poly)
    if abs(new_poly[0][0]-new_poly[-1][0]) > 1e-9 or abs(new_poly[0][1]-new_poly[-1][1]) > 1e-9:
        new_poly.append(new_poly[0])

    if _has_self_intersections(new_poly):
        return verts, {"radius_units": {}, "quarters_placed": 0, "semis_placed": 0}
    if _has_vertex_retouch(new_poly):
        return verts, {"radius_units": {}, "quarters_placed": 0, "semis_placed": 0}

    info = {
        "radius_units": {
            "quarter": {},
            "semi":    {},
        },
        "quarters_placed": len(chosen_quarters),
        "semis_placed":    len(chosen_semis),
    }
    # hist tiene sentido si hubo variedad
    return new_poly, info

# ───────── Generación base con intentos + relajación a mitad ─────────

def _generate_valid_contour(effective_seed:int, params:ContourParams,
                            quarters_want:int, semis_want:int,
                            attempts:int=32,
                            trace: Optional[List[Dict[str,Any]]] = None) -> Tuple[List[Tuple[float,float]], int, bool, Dict, Dict]:
    cs = params.cell_size
    gw = params.nx - 1
    gh = params.ny - 1
    want_any_arc = (quarters_want > 0) or (semis_want > 0)

    tries = [effective_seed] + [_derived_seed(effective_seed, k) for k in range(attempts)]
    arc_info_final: Dict = {"radius_units": {}, "quarters_placed": 0, "semis_placed": 0}
    reject_stats: DefaultDict[str,int] = defaultdict(int)

    # Copias mutables (para poder relajar)
    smin_local = params.min_straights_after_corner
    q_want_local = quarters_want
    s_want_local = semis_want
    relaxed = False

    for idx, si in enumerate(tries):
        if trace is not None and idx == attempts//2 and not relaxed:
            # evaluar relajación a mitad de ciclo
            if want_any_arc and reject_stats.get("no_arcs_when_required", 0) >= max(1, idx//2):
                relaxed = True
                smin_local = max(1, smin_local - 1)
                if s_want_local > 0:
                    s_want_local = max(0, s_want_local - 1)  # bajar semis primero: suelen ser más restrictivos
                trace.append({"phase":"relax", "idx":idx, "new_smin":smin_local, "q":q_want_local, "s":s_want_local})

        verts = build_orthogonal_simple_contour(si, ContourParams(
            nx=params.nx, ny=params.ny, cell_size=params.cell_size,
            max_corners=params.max_corners, max_step=1, smooth_passes=0,
            allow_diagonals_45=params.allow_diagonals_45,
            max_diagonals=params.max_diagonals,
            min_straights_after_corner=smin_local,
            min_diags_after_diag=params.min_diags_after_diag,
            lock_diag_opposite=params.lock_diag_opposite,
        ))

        xmin = min(x for (x,_) in verts)
        ymin = min(y for (_,y) in verts)
        if abs(xmin) > 1e-9 or abs(ymin) > 1e-9:
            verts = _translate_vertices(verts, dx=-xmin, dy=-ymin)

        verts_arc, arc_info = _apply_arcs_multiR(
            verts, cs, si,
            quarters_want=q_want_local, semis_want=s_want_local,
            smin=max(1, smin_local)
        )

        if want_any_arc and (arc_info.get("quarters_placed",0) + arc_info.get("semis_placed",0) == 0):
            reject_stats["no_arcs_when_required"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"no_arcs_when_required"})
            continue

        if not _valid_four_connected(verts_arc, gw, gh, cs):
            reject_stats["not_four_connected"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"not_four_connected"})
            continue
        if not _min_straights_ok(verts_arc, cs, smin_local):
            reject_stats["min_straights_violation"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"min_straights_violation"})
            continue
        if _has_self_intersections(verts_arc):
            reject_stats["self_intersection"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"self_intersection"})
            continue
        if _has_vertex_retouch(verts_arc):
            reject_stats["vertex_touch"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"vertex_touch"})
            continue

        corners = _count_corners(verts_arc)
        min_corners = 6 if smin_local < 2 else 10
        if (arc_info.get("quarters_placed",0)+arc_info.get("semis_placed",0)) == 0 and corners <= min_corners:
            reject_stats["too_simple"] += 1
            if trace is not None:
                trace.append({"phase":"reject","idx":idx,"reason":"too_simple"})
            continue

        arc_info_final = arc_info
        debug = {"reject_stats": dict(reject_stats), "auto_relax": relaxed}
        return verts_arc, idx, False, arc_info_final, debug

    # emergencia: figura base
    e_verts = _emergency_non_rectangular(effective_seed, ContourParams(
        nx=params.nx, ny=params.ny, cell_size=params.cell_size,
        max_corners=params.max_corners, max_step=1, smooth_passes=0,
        allow_diagonals_45=params.allow_diagonals_45,
        max_diagonals=params.max_diagonals,
        min_straights_after_corner=smin_local,
        min_diags_after_diag=params.min_diags_after_diag,
        lock_diag_opposite=params.lock_diag_opposite,
    ))
    xmin = min(x for (x,_) in e_verts)
    ymin = min(y for (_,y) in e_verts)
    if abs(xmin) > 1e-9 or abs(ymin) > 1e-9:
        e_verts = _translate_vertices(e_verts, dx=-xmin, dy=-ymin)

    debug = {"reject_stats": dict(reject_stats), "auto_relax": relaxed}
    return e_verts, attempts, True, {"radius_units": {}, "quarters_placed": 0, "semis_placed": 0}, debug

# ───────── Búsqueda de “seed más cercana” ─────────

def _search_nearest_valid(base_effective_seed:int, params:ContourParams,
                          quarters_want:int, semis_want:int,
                          attempts:int,
                          max_offset:int=256,
                          trace: Optional[List[Dict[str,Any]]] = None) -> Optional[Tuple[List[Tuple[float,float]], int, int, Dict, Dict]]:
    for k in range(1, max_offset+1):
        for sign in (1, -1):
            off = sign * k
            eff = _mix64((base_effective_seed + off) & ((1<<64)-1))
            if trace is not None and k % 16 == 0 and sign == 1:
                trace.append({"phase":"offset_scan","k":k})
            verts, tries, emergency, arc_info, dbg = _generate_valid_contour(
                eff, params, quarters_want, semis_want, attempts=attempts, trace=trace
            )
            placed = arc_info.get("quarters_placed",0) + arc_info.get("semis_placed",0)
            want_any = (quarters_want>0 or semis_want>0)
            if not emergency and (not want_any or placed>0):
                dbg2 = dict(dbg)
                dbg2["seed_correction_used_offset"] = off
                return verts, tries, off, arc_info, dbg2
    return None

# ───────── Seed portátil: pack/unpack ─────────

def _b36_to_int(s: str) -> int:
    s = s.strip().lower()
    val = 0
    for ch in s:
        if '0' <= ch <= '9':
            d = ord(ch) - ord('0')
        elif 'a' <= ch <= 'z':
            d = 10 + (ord(ch) - ord('a'))
        else:
            continue
        val = val * 36 + d
    return val

def _pack_payload(nx:int, ny:int, max_corners:int, allow_diag:int, max_diags:int,
                  smin:int, minddiag:int, lock:int, q:int, s:int, offset:int) -> int:
    offset_bias = offset + 2048
    fields = [
        (nx, 9), (ny, 9), (max_corners, 8),
        (allow_diag, 1), (max_diags, 6),
        (smin, 4), (minddiag, 4),
        (lock, 1), (q, 5), (s, 5),
        (offset_bias, 12),
    ]
    out = 0
    for val, bits in fields:
        if val < 0: val = 0
        out = (out << bits) | (val & ((1<<bits)-1))
    return out

def _unpack_payload(p:int) -> Dict[str,int]:
    specs = [
        ("nx",9), ("ny",9), ("max_corners",8),
        ("allow_diag",1), ("max_diags",6),
        ("smin",4), ("minddiag",4),
        ("lock",1), ("q",5), ("s",5),
        ("offset_bias",12),
    ]
    out: Dict[str,int] = {}
    for name,bits in reversed(specs):
        mask = (1<<bits)-1
        out[name] = p & mask
        p >>= bits
    ordered: Dict[str,int] = {}
    for name,_ in specs:
        ordered[name] = out[name]
    ordered["offset"] = ordered.pop("offset_bias") - 2048
    return ordered

def _make_portable_seed(seed_user_int:int, params:ContourParams,
                        q:int, s:int, used_offset:int) -> str:
    payload = _pack_payload(
        params.nx, params.ny, params.max_corners,
        1 if params.allow_diagonals_45 else 0, params.max_diagonals,
        params.min_straights_after_corner, params.min_diags_after_diag,
        1 if params.lock_diag_opposite else 0,
        q, s, used_offset
    )
    return f"PS1-{int_to_base36(seed_user_int)}-{int_to_base36(payload)}"

def _parse_portable_seed(ps: str) -> Optional[Dict[str,int]]:
    ps = ps.strip()
    if not ps.lower().startswith("ps1-"):
        return None
    try:
        _, a, b = ps.split("-", 2)
        seed_user_int = _b36_to_int(a)
        payload = _b36_to_int(b)
        fields = _unpack_payload(payload)
        fields["seed_user_int"] = seed_user_int
        return fields
    except Exception:
        return None

# ───────── API ─────────

def generate_contour(
    seed: str | None = None,
    max_corners: int = 20,
    nx_range: Tuple[int, int] = (12, 20),
    ny_range: Tuple[int, int] = (12, 20),

    allow_diagonals_45: bool = False,
    max_diagonals: int = 10,
    min_straights_after_corner: int = 1,
    min_diags_after_diag: int = 2,
    lock_diag_opposite: bool = True,

    quarter_arcs_count: int | None = 0,
    semi_arcs_count: int | None = 0,
    holes_spec: Optional[Dict[str, int]] = None,

    portable_seed: str | None = None,

    attempts: int = 16,
    correction_max_offset: int = 128,
) -> Dict:
    trace: List[Dict[str,Any]] = []

    # ─── Modo seed portátil ───
    if portable_seed:
        parsed = _parse_portable_seed(portable_seed)
        if parsed:
            seed_user_int = parsed["seed_user_int"]
            nx = max(6, int(parsed["nx"]))
            ny = max(6, int(parsed["ny"]))
            cell_size = 12.0
            qwant = int(parsed["q"])
            swant = int(parsed["s"])
            used_offset = int(parsed["offset"])
            params = ContourParams(
                nx=nx, ny=ny, cell_size=cell_size,
                max_corners=int(parsed["max_corners"]),
                max_step=1, smooth_passes=0,
                allow_diagonals_45=bool(parsed["allow_diag"]),
                max_diagonals=int(parsed["max_diags"]) if bool(parsed["allow_diag"]) else 0,
                min_straights_after_corner=int(max(1, parsed["smin"])),
                min_diags_after_diag=int(max(1, parsed["minddiag"])),
                lock_diag_opposite=bool(parsed["lock"]),
            )

            # Huecos
            if holes_spec is None:
                params.random_holes = True
                params.num_circles = 0
                params.num_triangles = 0
                params.num_quads = 0
            else:
                params.random_holes = False
                params.num_circles = int(holes_spec.get("circles", 0))
                params.num_triangles = int(holes_spec.get("triangles", 0))
                params.num_quads = int(holes_spec.get("quads", 0))

            fp = _fingerprint_exact_params(params, qwant, swant)
            base_eff = _mix64(seed_user_int ^ fp)
            eff = _mix64((base_eff + used_offset) & ((1<<64)-1))

            verts, tries_used, emergency, arc_info, debug = _generate_valid_contour(
                eff, params, quarters_want=qwant, semis_want=swant, attempts=attempts, trace=trace
            )

            # Huecos reproducibles
            rng_holes = RNG(eff ^ 0xC0FFEE)
            holes_pts, holes_meta = _generate_holes(params, rng_holes, verts)

            base = (nx - 1) * cell_size
            height = (ny - 1) * cell_size

            # Centroide compuesto (contorno – huecos)
            _, Cx, Cy = _composite_area_centroid_exact(verts, holes_meta)
            cx_px, cy_px = (Cx, Cy)
            cx_u, cy_u = cx_px / cell_size, cy_px / cell_size

            debug["trace"] = trace[:128]
            exercise: Dict = {
                "seed": int_to_base36(seed_user_int),
                "seed_effective": int_to_base36(eff),
                "config_sig": int_to_base36(fp),
                "generator_version": GEN_VERSION_CONTOUR,
                "portable_seed": _make_portable_seed(seed_user_int, params, qwant, swant, used_offset),
                "portable_mode": True,
                "grid": {"cell_size": cell_size, "width": nx - 1, "height": ny - 1},
                "pieces": [],
                "contour": {"vertices": verts},
                "holes": holes_pts,
                "holes_meta": holes_meta,
                "bbox": {"xmin": 0.0, "xmax": base, "ymin": 0.0, "ymax": height},
                "measures": {"base": base, "height": height},
                "circular_features": [
                    {"type": "quarter_arc", "count": qwant, "placed": arc_info.get("quarters_placed", 0)},
                    {"type": "semi_arc",    "count": swant, "placed": arc_info.get("semis_placed", 0)},
                ],
                "solution": {"cx": cx_px, "cy": cy_px},
                "solution_units": {"ux": cx_u, "uy": cy_u},
                "debug": {
                    "tries_used": tries_used,
                    "emergency_fallback": emergency,
                    "arc_radii_hist": arc_info.get("radius_units", {}),
                    "reject_stats": debug.get("reject_stats", {}),
                    "status": "ok" if not emergency else "emergency_fallback",
                    "warnings": (["emergency_fallback_used"] if emergency else []),
                    "seed_correction": {"used": False, "offset": used_offset,
                                        "from_effective": int_to_base36(base_eff),
                                        "to_effective": int_to_base36(eff)},
                    "trace": debug.get("trace", []),
                    "auto_relax": debug.get("auto_relax", False),
                },
            }
            return exercise

    # ─── Modo normal ───
    rng = RNG(seed)
    seed_user_int = rng.seed_int

    nx = max(6, rng.randint(nx_range[0], nx_range[1]))
    ny = max(6, rng.randint(ny_range[0], ny_range[1]))
    cell_size = 12.0

    quarters_want = 0 if quarter_arcs_count is None else int(max(0, quarter_arcs_count))
    semis_want    = 0 if semi_arcs_count    is None else int(max(0, semi_arcs_count))
    if quarter_arcs_count is None:
        quarters_want = rng.randint(0, 3)
    if semi_arcs_count is None:
        semis_want = rng.randint(0, 2)

    params = ContourParams(
        nx=nx, ny=ny, cell_size=cell_size,
        max_corners=max_corners,
        max_step=1, smooth_passes=0,
        allow_diagonals_45=allow_diagonals_45,
        max_diagonals=max_diagonals if allow_diagonals_45 else 0,
        min_straights_after_corner=int(max(1, min_straights_after_corner)),
        min_diags_after_diag=int(max(1, min_diags_after_diag)),
        lock_diag_opposite=bool(lock_diag_opposite),
    )

    if holes_spec is None:
        params.random_holes = True
        params.num_circles = 0
        params.num_triangles = 0
        params.num_quads = 0
    else:
        params.random_holes = False
        params.num_circles = int(holes_spec.get("circles", 0))
        params.num_triangles = int(holes_spec.get("triangles", 0))
        params.num_quads = int(holes_spec.get("quads", 0))

    fp = _fingerprint_exact_params(params, quarters_want, semis_want)
    base_eff = _mix64(seed_user_int ^ fp)

    verts, tries_used, emergency, arc_info, debug = _generate_valid_contour(
        base_eff, params, quarters_want=quarters_want, semis_want=semis_want, attempts=attempts, trace=trace
    )

    corrected = False
    used_offset = 0
    if (emergency or (quarters_want+semis_want>0 and arc_info.get("quarters_placed",0)+arc_info.get("semis_placed",0)==0)):
        res = _search_nearest_valid(base_eff, params, quarters_want, semis_want,
                                    attempts=attempts, max_offset=correction_max_offset, trace=trace)
        if res is not None:
            verts, tries_used2, used_offset, arc_info, debug2 = res
            tries_used = tries_used2
            emergency = False
            corrected = True
            for k,v in debug2.items():
                debug[k] = v

    eff = _mix64((base_eff + used_offset) & ((1<<64)-1))
    base = (nx - 1) * cell_size
    height = (ny - 1) * cell_size

    # Huecos
    rng_holes = RNG(eff ^ 0xC0FFEE)
    holes_pts, holes_meta = _generate_holes(params, rng_holes, verts)

    # Centroide compuesto (contorno – huecos)
    _, Cx, Cy = _composite_area_centroid_exact(verts, holes_meta)
    cx_px, cy_px = (Cx, Cy)
    cx_u, cy_u = cx_px / cell_size, cy_px / cell_size

    warnings: List[str] = []
    if emergency:
        warnings.append("emergency_fallback_used")
    placed_q = arc_info.get("quarters_placed", 0)
    placed_s = arc_info.get("semis_placed", 0)
    if placed_q < quarters_want:
        warnings.append("quarters_partially_placed")
    if placed_s < semis_want:
        warnings.append("semis_partially_placed")

    corners = _count_corners(verts)
    min_corners_thr = 6 if params.min_straights_after_corner < 2 else 10
    if corners <= (min_corners_thr + 1) and (placed_q + placed_s) <= 1:
        warnings.append("low_complexity_shape")

    debug["trace"] = trace[:128]
    exercise: Dict = {
        "seed": int_to_base36(seed_user_int),
        "seed_effective": int_to_base36(eff),
        "config_sig": int_to_base36(fp),
        "generator_version": GEN_VERSION_CONTOUR,
        "portable_seed": _make_portable_seed(seed_user_int, params, quarters_want, semis_want, used_offset),
        "portable_mode": False,
        "grid": {"cell_size": cell_size, "width": nx - 1, "height": ny - 1},
        "pieces": [],
        "contour": {"vertices": verts},
        "holes": holes_pts,
        "holes_meta": holes_meta,
        "bbox": {"xmin": 0.0, "xmax": base, "ymin": 0.0, "ymax": height},
        "measures": {"base": base, "height": height},
        "circular_features": [
            {"type": "quarter_arc", "count": quarters_want, "placed": placed_q},
            {"type": "semi_arc",    "count": semis_want,    "placed": placed_s},
        ],
        "solution": {"cx": cx_px, "cy": cy_px},
        "solution_units": {"ux": cx_u, "uy": cy_u},
        "debug": {
            "tries_used": tries_used,
            "emergency_fallback": emergency,
            "arc_radii_hist": arc_info.get("radius_units", {}),
            "reject_stats": debug.get("reject_stats", {}),
            "status": "ok" if not emergency else "emergency_fallback",
            "warnings": warnings,
            "corners": corners,
            "min_corners_threshold": min_corners_thr,
            "seed_correction": {
                "used": corrected,
                "offset": used_offset,
                "from_effective": int_to_base36(base_eff),
                "to_effective": int_to_base36(eff),
            },
            "trace": debug.get("trace", []),
            "auto_relax": debug.get("auto_relax", False),
        },
    }
    return exercise

