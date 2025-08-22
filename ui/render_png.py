from __future__ import annotations
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def exercise_to_png_bytes(
    ex: Dict,
    margin: float = 30.0,
    show_centroid: bool = False,
    scale: int = 3,
    fill_color: str = "#222222",
    stroke_color: str = "#000000",
    axis_color: str = "#0a66c2",
    grid_color: str = "#0a66c2",
    show_grid: bool = False,
    show_grid_top: bool = False,
    grid_top_color: str = "#ffffff",
) -> bytes:
    from io import BytesIO

    grid = ex["grid"]
    cs   = grid["cell_size"]
    gw   = grid["width"]
    gh   = grid["height"]

    base_grid   = gw * cs
    height_grid = gh * cs

    W = int((base_grid + 2*margin) * scale)
    H = int((height_grid + 2*margin) * scale)

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    def rgb(hexstr: str):
        hexstr = hexstr.lstrip("#")
        return tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))

    axis_rgb   = rgb(axis_color)
    fill_rgb   = rgb(fill_color)
    stroke_rgb = rgb(stroke_color)
    grid_rgb   = rgb(grid_color)
    grid_top_rgb = rgb(grid_top_color)

    origin = (int(margin*scale), int((margin+height_grid)*scale))

    # Ejes
    draw.line([origin, (int((margin+base_grid+10)*scale), origin[1])], fill=axis_rgb, width=max(1,scale))
    draw.line([origin, (origin[0], int((margin-10)*scale))], fill=axis_rgb, width=max(1,scale))
    
    # Texto de ejes (dibujado al tamaño final para evitar blur)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Grilla debajo
    if show_grid:
        for i in range(gw + 1):
            x = int((margin + i*cs) * scale)
            draw.line([(x, origin[1]), (x, int((margin)*scale))], fill=grid_rgb, width=1)
        for j in range(gh + 1):
            y = int((margin + height_grid - j*cs) * scale)
            draw.line([(origin[0], y), (int((margin+base_grid)*scale), y)], fill=grid_rgb, width=1)

    # Contorno principal
    pts = []
    if "contour" in ex and ex["contour"].get("vertices"):
        for (x,y) in ex["contour"]["vertices"]:
            sx = int((margin + x) * scale)
            sy = int((margin + (height_grid - y)) * scale)
            pts.append((sx, sy))
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        if pts:
            # Dibujar el relleno
            draw.polygon(pts, fill=fill_rgb)
            # Dibujar explícitamente los bordes
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i+1]], fill=stroke_rgb, width=max(2, scale//2))

    # Huecos (polígonos blancos con borde de stroke)
    for hole in ex.get("holes", []):
        hpts: List[Tuple[int,int]] = []
        for (x,y) in hole:
            sx = int((margin + x) * scale)
            sy = int((margin + (height_grid - y)) * scale)
            hpts.append((sx, sy))
        if hpts:
            if hpts[0] != hpts[-1]:
                hpts.append(hpts[0])
            # Dibujar el relleno blanco
            draw.polygon(hpts, fill=(255,255,255))
            # Dibujar explícitamente los bordes
            for i in range(len(hpts) - 1):
                draw.line([hpts[i], hpts[i+1]], fill=stroke_rgb, width=max(2, scale//2))

    # Grilla superior (solo sobre la figura sólida)
    if show_grid_top:
        # Crear una imagen de máscara
        mask_img = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        
        # Dibujar el contorno principal en la máscara (con grosor para incluir bordes)
        if "contour" in ex and ex["contour"].get("vertices"):
            contour_pts = []
            for (x,y) in ex["contour"]["vertices"]:
                sx = int((margin + x) * scale)
                sy = int((margin + (height_grid - y)) * scale)
                contour_pts.append((sx, sy))
            if contour_pts and contour_pts[0] != contour_pts[-1]:
                contour_pts.append(contour_pts[0])
            if contour_pts:
                # Dibujar con grosor para incluir los bordes
                mask_draw.polygon(contour_pts, fill=255)
                # Agregar grosor adicional para los bordes
                for i in range(len(contour_pts) - 1):
                    mask_draw.line([contour_pts[i], contour_pts[i+1]], fill=255, width=max(2, scale//2))
        
        # Restar los huecos de la máscara
        for hole in ex.get("holes", []):
            hpts = []
            for (x,y) in hole:
                sx = int((margin + x) * scale)
                sy = int((margin + (height_grid - y)) * scale)
                hpts.append((sx, sy))
            if hpts:
                if hpts[0] != hpts[-1]:
                    hpts.append(hpts[0])
                # Restar el relleno del hueco
                mask_draw.polygon(hpts, fill=0)
                # Restar también los bordes del hueco
                for i in range(len(hpts) - 1):
                    mask_draw.line([hpts[i], hpts[i+1]], fill=0, width=max(2, scale//2))
        
        # Convertir máscara a array numpy para verificación rápida
        mask_array = np.array(mask_img)
        
        # Dibujar líneas de grilla solo donde la máscara sea 255 (figura sólida)
        for i in range(gw + 1):
            x = int((margin + i*cs) * scale)
            # Verificar cada segmento de línea vertical
            for j in range(gh + 1):
                y_start = int((margin + height_grid - j*cs) * scale)
                y_end = int((margin + height_grid - (j+1)*cs) * scale)
                # Verificar si el punto medio del segmento está en la figura sólida
                mid_y = (y_start + y_end) // 2
                if 0 <= mid_y < H and mask_array[mid_y, x] == 255:
                    # Dibujar segmento de línea a guiones
                    segment_length = int(cs * scale / 4)
                    gap_length = int(cs * scale / 4)
                    current_y = y_start
                    while current_y > y_end:
                        next_y = max(current_y - segment_length, y_end)
                        draw.line([(x, current_y), (x, next_y)], fill=grid_top_rgb, width=max(1, scale//6))
                        current_y = next_y - gap_length
                        if current_y <= y_end:
                            break
        
        for j in range(gh + 1):
            y = int((margin + height_grid - j*cs) * scale)
            # Verificar cada segmento de línea horizontal
            for i in range(gw + 1):
                x_start = int((margin + i*cs) * scale)
                x_end = int((margin + (i+1)*cs) * scale)
                # Verificar si el punto medio del segmento está en la figura sólida
                mid_x = (x_start + x_end) // 2
                if 0 <= mid_x < W and mask_array[y, mid_x] == 255:
                    # Dibujar segmento de línea a guiones
                    segment_length = int(cs * scale / 4)
                    gap_length = int(cs * scale / 4)
                    current_x = x_start
                    while current_x < x_end:
                        next_x = min(current_x + segment_length, x_end)
                        draw.line([(current_x, y), (next_x, y)], fill=grid_top_rgb, width=max(1, scale//6))
                        current_x = next_x + gap_length
                        if current_x >= x_end:
                            break

    # Reescalar solo si es necesario (mantener alta resolución)
    if scale != 1:
        # No reescalar de vuelta a 1x - mantener la alta resolución
        # Solo redibujar texto de ejes en el tamaño final
        draw = ImageDraw.Draw(img)
        # Recalcular posiciones para el tamaño final (con scale aplicado)
        final_x_pos = int((margin+base_grid+15)*scale)
        final_y_pos = int((margin+5+height_grid)*scale)
        final_y_label_pos = int((margin-10)*scale)
        final_y_label_y = int((margin-15)*scale)
        draw.text((final_x_pos, final_y_pos), "X", fill=axis_rgb, font=font)
        draw.text((final_y_label_pos, final_y_label_y), "Y", fill=axis_rgb, font=font)
    else:
        # Si no hay escalado, dibujar texto directamente
        draw.text((int((margin+base_grid+15)*scale), int((margin+5+height_grid)*scale)), "X", fill=axis_rgb, font=font)
        draw.text((int((margin-10)*scale), int((margin-15)*scale)), "Y", fill=axis_rgb, font=font)

    # Centroide (dibujado al final para asegurar visibilidad)
    if show_centroid and "solution" in ex:
        cx = ex["solution"]["cx"]; cy = ex["solution"]["cy"]
        cxp = int((margin + cx) * scale)
        cyp = int((margin + (height_grid - cy)) * scale)
        r = 3 * scale
        draw.ellipse([cxp-r, cyp-r, cxp+r, cyp+r], fill=(255,0,0))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
