from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class ContourParams:
    nx: int                 # puntos en X (>= 6)
    ny: int                 # puntos en Y (>= 6)
    cell_size: float        # tamaño de celda

    # Complejidad / calidad
    max_corners: int = 20                   # límite superior deseado de "esquinas" (cambios de dirección 90°)
    max_step: int = 1                       # sin uso por ahora
    smooth_passes: int = 0                  # sin uso por ahora

    # Diagonales 45°
    allow_diagonals_45: bool = False
    max_diagonals: int = 0                  # número máximo de aristas diagonales a colocar
    min_diags_after_diag: int = 2           # largo mínimo (en celdas) de una arista diagonal
    lock_diag_opposite: bool = True         # evita diagonales consecutivas con signos opuestos

    # Reglas locales
    min_straights_after_corner: int = 1     # mínimo de celdas rectas entre giros de 90°

    # Huecos internos
    random_holes: bool = False              # si es True, se ignoran las cantidades específicas
    num_circles: int = 0                    # cantidad de huecos circulares
    num_triangles: int = 0                  # cantidad de huecos triangulares
    num_quads: int = 0                      # cantidad de huecos cuadriláteros

# Utilidades en grilla (unidad = 1 celda)
def _segment_len(p:Tuple[int,int], q:Tuple[int,int]) -> int:
    return max(abs(q[0]-p[0]), abs(q[1]-p[1]))

def _dir_vector(p,q):
    dx, dy = q[0]-p[0], q[1]-p[1]
    if dx>0 and dy==0: return (1,0)
    if dx<0 and dy==0: return (-1,0)
    if dy>0 and dx==0: return (0,1)
    if dy<0 and dx==0: return (0,-1)
    if dx==dy and dx>0: return (1,1)   # NE
    if dx==dy and dx<0: return (-1,-1) # SW
    if dx==-dy and dx>0: return (1,-1) # SE
    if dx==-dy and dx<0: return (-1,1) # NW
    return (dx,dy)

def _simplify_colinear(poly: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if len(poly)<=3: return poly
    out=[poly[0]]
    for i in range(1,len(poly)-1):
        a=out[-1]; b=poly[i]; c=poly[i+1]
        vab=_dir_vector(a,b); vbc=_dir_vector(b,c)
        if vab==vbc:  # colineales
            continue
        out.append(b)
    out.append(poly[-1])
    return out

def _diag_sign(dir_prev:Tuple[int,int], dir_next:Tuple[int,int]) -> int:
    mapping = {
        ((1,0),(0,1)): +1,  ((0,1),(-1,0)): +1,  ((-1,0),(0,-1)): +1,  ((0,-1),(1,0)): +1,   # CCW
        ((1,0),(0,-1)): -1, ((0,-1),(-1,0)): -1, ((-1,0),(0,1)): -1,  ((0,1),(1,0)): -1,    # CW
    }
    return mapping.get((dir_prev, dir_next), +1)

def _offset_point(p:Tuple[int,int], dirv:Tuple[int,int], k:int)->Tuple[int,int]:
    return (p[0]+dirv[0]*k, p[1]+dirv[1]*k)

def _is_axis_unit(v: Tuple[int,int]) -> bool:
    return (abs(v[0]) == 1 and v[1] == 0) or (abs(v[1]) == 1 and v[0] == 0)

def _insert_corner_diagonals(poly: List[Tuple[int,int]], params: ContourParams) -> List[Tuple[int,int]]:
    """Reemplaza esquinas ortogonales por diagonales de 45° respetando las reglas del usuario."""
    if not params.allow_diagonals_45 or params.max_diagonals <= 0:
        return poly

    n = len(poly)
    if n < 4:
        return poly

    # Asegurar polígono cerrado
    closed = (poly[0] == poly[-1])
    work = poly[:-1] if closed else poly[:]

    max_diags = int(max(0, params.max_diagonals))
    k_min = int(max(1, params.min_diags_after_diag))
    lock_opp = bool(params.lock_diag_opposite)

    placed = 0
    out: List[Tuple[int,int]] = []
    last_sign: int | None = None

    for i in range(len(work)):
        a = work[i-1]
        b = work[i]
        c = work[(i+1) % len(work)]

        vab = _dir_vector(a, b)
        vbc = _dir_vector(b, c)

        # Solo tratamos esquinas ortogonales (ejes) -> (ejes)
        if _is_axis_unit(vab) and _is_axis_unit(vbc) and (vab != vbc) and placed < max_diags:
            # Longitudes disponibles hasta la esquina b
            L1 = _segment_len(a, b)
            L2 = _segment_len(b, c)

            # Necesitamos al menos 1 celda a cada lado para que tenga sentido,
            # y al menos k_min para cumplir la regla del usuario.
            k = min(k_min, L1-1, L2-1)
            if k >= 1:
                # Chequeo de signo para evitar zig-zag inmediato
                sign = _diag_sign(vab, vbc)  # +1 CCW, -1 CW
                if not (lock_opp and (last_sign is not None) and (sign == -last_sign)):
                    # Construir p y q recortando k celdas desde la esquina
                    p = (b[0] - vab[0]*k, b[1] - vab[1]*k)
                    q = (b[0] + vbc[0]*k, b[1] + vbc[1]*k)

                    # Evitar duplicados si a==p o q==c por redondeos raros (no debería)
                    if not (p == b or q == b or p == q):
                        # Emitimos 'a' (si es el primero o cambia)
                        if not out or out[-1] != a:
                            out.append(a)
                        # tramo a->p (eje)
                        if out[-1] != p:
                            out.append(p)
                        # tramo diagonal p->q
                        out.append(q)
                        last_sign = sign
                        placed += 1
                        # saltamos agregar 'b' (esquina original), y dejamos que el loop
                        # agregue 'c' normalmente en la siguiente iteración
                        continue

        # Caso sin diagonal: emitir vértice normal
        if not out or out[-1] != b:
            out.append(b)

    # Cerrar polígono
    if closed:
        if out[0] != out[-1]:
            out.append(out[0])
    return _simplify_colinear(out)

def build_orthogonal_with_notches_and_diagonals(seed_int:int, params:ContourParams) -> List[Tuple[float,float]]:
    rng = random.Random(seed_int)

    # Área utilizable (en vértices de grilla)
    w = params.nx - 1
    h = params.ny - 1
    cs = params.cell_size
    margin = 2
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    W = x1 - x0
    H = y1 - y0
    if W<6 or H<6:
        verts = [(x0*cs,y0*cs),(x1*cs,y0*cs),(x1*cs,y1*cs),(x0*cs,y1*cs),(x0*cs,y0*cs)]
        return verts

    # meta de esquinas: base 4 + 4 por muesca
    max_corners = max(4, params.max_corners)
    requested_notches = max(0, (max_corners - 4)//4)

    # distribuir muescas por borde
    per_edge = [0,0,0,0]  # bottom, right, top, left
    for _ in range(requested_notches):
        per_edge[rng.randrange(4)] += 1

    smin = max(1, params.min_straights_after_corner)

    def place_notches(edge_len:int, count:int, smin:int, depth_max:int) -> List[Tuple[int,int,int]]:
        if count<=0:
            return []
        usable = max(0, edge_len - 2*smin)   # dejamos smin al inicio y al final
        if usable <= 0:
            return []
        # anchos y posiciones no solapadas con separación mínima smin
        taken = []
        tries = 0
        while len(taken)<count and tries<2000:
            tries += 1
            inner_len = rng.randint(smin, max(smin, max(1, usable//max(1,count))))
            start = rng.randint(0, max(0, usable-inner_len))
            start += smin
            ok=True
            for a,b in taken:
                if not (start+inner_len+smin <= a or b+smin <= start):
                    ok=False; break
            if ok:
                taken.append((start, start+inner_len))
        taken.sort()
        out=[]
        for (a,b) in taken:
            inner_len = b-a
            depth = rng.randint(smin, max(smin, max(1, depth_max)))
            out.append((a, inner_len, depth))
        if not out and count>0 and usable>0:
            inner_len = max(smin, min(usable, max(1, usable//2)))
            start = smin + max(0, (usable - inner_len)//2)
            depth = max(smin, max(1, depth_max//2))
            out.append((start, inner_len, depth))
        return out

    bottom_notches = place_notches(W, per_edge[0], smin, H//3)
    right_notches  = place_notches(H, per_edge[1], smin, W//3)
    top_notches    = place_notches(W, per_edge[2], smin, H//3)
    left_notches   = place_notches(H, per_edge[3], smin, W//3)

    # Construcción CCW con muescas hacia adentro
    poly: List[Tuple[int,int]] = []

    # BOTTOM
    curx = x0
    poly.append((x0,y0))
    for (start, inner_len, depth) in bottom_notches:
        sx = max(curx + smin, x0 + start)
        ex = min(x1 - smin, sx + inner_len)
        if ex <= sx: continue
        poly.append((sx, y0))
        poly.append((sx, y0+depth))
        poly.append((ex, y0+depth))
        poly.append((ex, y0))
        curx = ex
    poly.append((x1, y0))

    # RIGHT
    cury = y0
    for (start, inner_len, depth) in right_notches:
        sy = max(cury + smin, y0 + start)
        ey = min(y1 - smin, sy + inner_len)
        if ey <= sy: continue
        poly.append((x1, sy))
        poly.append((x1 - depth, sy))
        poly.append((x1 - depth, ey))
        poly.append((x1, ey))
        cury = ey
    poly.append((x1, y1))

    # TOP
    curx = x1
    for (start, inner_len, depth) in top_notches:
        sx = min(curx - smin, x1 - start)
        ex = max(x0 + smin, sx - inner_len)
        if ex >= sx: continue
        poly.append((sx, y1))
        poly.append((sx, y1 - depth))
        poly.append((ex, y1 - depth))
        poly.append((ex, y1))
        curx = ex
    poly.append((x0, y1))

    # LEFT
    cury = y1
    for (start, inner_len, depth) in left_notches:
        sy = min(cury - smin, y1 - start)
        ey = max(y0 + smin, sy - inner_len)
        if ey >= sy: continue
        poly.append((x0, sy))
        poly.append((x0 + depth, sy))
        poly.append((x0 + depth, ey))
        poly.append((x0, ey))
        cury = ey
    poly.append((x0, y0))

    # Simplificar colineales
    poly = _simplify_colinear(poly)

    # >>> NUEVO: insertar diagonales en esquinas conforme a parámetros
    poly = _insert_corner_diagonals(poly, params)
    # <<<

    if poly[0] != poly[-1]:
        poly.append(poly[0])

    verts = [(x*cs, y*cs) for (x,y) in poly]
    return verts

# API pública usada por el generador
def build_orthogonal_simple_contour(seed_int:int, params:ContourParams) -> List[Tuple[float,float]]:
    return build_orthogonal_with_notches_and_diagonals(seed_int, params)
