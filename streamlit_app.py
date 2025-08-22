import streamlit as st
from core.generator_contour import generate_contour, GEN_VERSION_CONTOUR
from ui.render_svg import exercise_to_svg
from ui.render_png import exercise_to_png_bytes
import streamlit.components.v1 as components

st.set_page_config(page_title="Física I — Centro de Gravedad", page_icon="📐", layout="wide")

DEFAULTS = {
    # Grilla
    "nx_min": 12, "nx_max": 20,
    "ny_min": 12, "ny_max": 20,
    # Reglas de trazado
    "max_corners": 20,
    "max_diagonals": 10,
    "min_straights_after_corner": 2,   # por pedido: default 2
    "min_diags_after_diag": 2,
    # Arcos
    "quarters_random": True, "quarters_want": 0,
    "semis_random": True,    "semis_want": 0,
    # Huecos
    "holes_random": True,
    "holes_circles": 0,
    "holes_triangles": 0,
    "holes_quads": 0,
    # Estilo
    "fill_color": "#808080",
    "stroke_color": "#000000",
    "axis_color": "#0a66c2",
    "grid_color": "#0a66c2",
    "grid_top_color": "#ffffff",
    # Vista
    "show_grid": False,
    "show_grid_top": False,
    "show_centroid": False,
    # Avanzado (performance)
    "fast_mode": False,
    "attempts": 16,
    "correction_max_offset": 128,
    # Seed PS1
    "portable_seed_str": "",
}

def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

if "initialized" not in st.session_state:
    reset_defaults()
    st.session_state["initialized"] = True
    st.session_state["view"] = "exercise"

st.title("Generador de Contornos — Centro de Gravedad")

# ───────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.header("Controles")

    if st.button("Restablecer valores por defecto"):
        reset_defaults()

    # Semilla
    with st.expander("Semilla", expanded=False):
        portable_seed_str = st.text_input(
            "Seed",
            value=st.session_state.get("portable_seed_str", ""),
            key="portable_seed_str_input",
            help="Pegá acá una seed para reproducir exactamente la misma figura.",
            placeholder="Seed..."
        )
        use_portable = portable_seed_str.strip().lower().startswith("ps1-")

    # Grilla con sliders dobles (mín–máx) — límites 6–50
    with st.expander("Grilla", expanded=False):
        nx_min, nx_max = st.slider(
            "Puntos en X (rango)",
            min_value=6, max_value=50,
            value=(st.session_state["nx_min"], st.session_state["nx_max"]),
            step=1, key="nx_range_slider",
            help="Arrastrá los dos puntos para fijar mínimo y máximo."
        )
        ny_min, ny_max = st.slider(
            "Puntos en Y (rango)",
            min_value=6, max_value=50,
            value=(st.session_state["ny_min"], st.session_state["ny_max"]),
            step=1, key="ny_range_slider",
            help="Arrastrá los dos puntos para fijar mínimo y máximo."
        )

    # Reglas del trazado
    with st.expander("Reglas de trazado", expanded=False):
        cR1, cR2 = st.columns(2)
        with cR1:
            max_corners = st.number_input(
                "Máx. esquinas",
                min_value=4, max_value=400,
                value=st.session_state["max_corners"],
                step=1, key="max_corners_input",
            )
            min_straights_after_corner = st.number_input(
                "Rectas tras esquina (mín.)",
                min_value=1, max_value=50,
                value=st.session_state["min_straights_after_corner"],
                step=1, key="min_straights_input",
                help="Celdas rectas mínimas después de girar 90° (default 2).",
            )
        with cR2:
            max_diagonals = st.number_input(
                "Máx. diagonales 45°",
                min_value=0, max_value=400,
                value=st.session_state["max_diagonals"],
                step=1, key="max_diags_input",
                help="0 desactiva diagonales; >0 habilita hasta ese máximo.",
            )
            min_diags_after_diag = st.number_input(
                "Diagonales tras diagonal (mín.)",
                min_value=1, max_value=50,
                value=st.session_state["min_diags_after_diag"],
                step=1, key="min_diags_input",
                help="Largo mínimo (en celdas) de cada diagonal (default 2).",
            )

    # Arcos
    with st.expander("Arcos", expanded=False):
        cA1, cA2 = st.columns(2)
        with cA1:
            quarters_random = st.checkbox(
                "Cuartos de círculo aleatorios",
                value=st.session_state["quarters_random"],
                key="quarters_random_cb"
            )
            quarters_want = st.number_input(
                "Cantidad (cuartos)",
                min_value=0, max_value=20,
                value=st.session_state["quarters_want"],
                step=1, key="quarters_want_input",
                disabled=quarters_random,
                help="0 = ninguno. Si 'aleatorios' está activo, este valor se ignora."
            )
        with cA2:
            semis_random = st.checkbox(
                "Semicircunferencias aleatorias",
                value=st.session_state["semis_random"],
                key="semis_random_cb"
            )
            semis_want = st.number_input(
                "Cantidad (semis)",
                min_value=0, max_value=20,
                value=st.session_state["semis_want"],
                step=1, key="semis_want_input",
                disabled=semis_random,
                help="0 = ninguno. Si 'aleatorias' está activo, este valor se ignora."
            )

    # Huecos
    with st.expander("Huecos (blancos)", expanded=False):
        holes_random = st.checkbox(
            "Huecos aleatorios",
            value=st.session_state.get("holes_random", True),
            key="holes_random_cb"
        )
        cH1, cH2, cH3 = st.columns(3)
        with cH1:
            holes_circles = st.number_input(
                "Círculos",
                min_value=0, max_value=50,
                value=st.session_state.get("holes_circles", 0),
                step=1, key="holes_circles_input",
                disabled=holes_random
            )
        with cH2:
            holes_triangles = st.number_input(
                "Triángulos",
                min_value=0, max_value=50,
                value=st.session_state.get("holes_triangles", 0),
                step=1, key="holes_triangles_input",
                disabled=holes_random
            )
        with cH3:
            holes_quads = st.number_input(
                "Cuadriláteros",
                min_value=0, max_value=50,
                value=st.session_state.get("holes_quads", 0),
                step=1, key="holes_quads_input",
                disabled=holes_random
            )
        st.caption("Poné 0 para omitir; si 'aleatorios' está activo, estas cantidades se ignoran.")

    # Avanzado (performance)
    with st.expander("Avanzado", expanded=False):
        fast_mode = st.checkbox("Modo rápido (menos intentos)", value=st.session_state["fast_mode"], key="fast_mode_cb")
        attempts = st.number_input(
            "Intentos base por seed",
            min_value=4, max_value=64,
            value=(8 if fast_mode else st.session_state["attempts"]),
            step=1, key="attempts_input",
            help="Sub-semillas a probar por cada seed efectiva."
        )
        correction_max_offset = st.number_input(
            "Búsqueda de seed cercana (±offset)",
            min_value=0, max_value=512,
            value=(32 if fast_mode else st.session_state["correction_max_offset"]),
            step=1, key="offset_input",
            help="Radio de búsqueda para corregir la seed cuando faltan arcos o falla alguna validación."
        )
        st.caption("Tip: si tu PC está justa, activá Modo rápido y mantené grillas moderadas.")

    # Diagnóstico de conectividad (JS simple)
    with st.expander("Diagnóstico (conectividad del navegador)", expanded=False):
        components.html(
            """
            <div id="net-status" style="font-family:system-ui,Arial; font-size:13px;">
              <strong>Estado:</strong> <span id="net-result">Chequeando…</span>
            </div>
            <script>
              async function ping() {
                try {
                  // Ping liviano (no-cors): si resuelve, consideramos que hay red
                  await fetch('https://www.gstatic.com/generate_204', {mode:'no-cors', cache:'no-store'});
                  document.getElementById('net-result').textContent = 'OK (online)';
                } catch (e) {
                  document.getElementById('net-result').textContent = 'SIN CONEXIÓN';
                  console.warn('Ping de conectividad falló:', e);
                }
              }
              ping();
              // re-chequear cada 20 s
              setInterval(ping, 20000);
              console.log('[CG UI] componente de conectividad montado');
            </script>
            """,
            height=40
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Generar**")
    gen_btn = st.button("Generar ejercicio", type="primary", use_container_width=True)
    manual_btn = st.button("Manual de usuario", use_container_width=True)

# Persistir lo elegido al presionar
if gen_btn:
    st.session_state["view"] = "exercise"
    st.session_state.update({
        "portable_seed_str": portable_seed_str,
        "nx_min": nx_min, "nx_max": nx_max, "ny_min": ny_min, "ny_max": ny_max,
        "max_corners": max_corners, "max_diagonals": max_diagonals,
        "min_straights_after_corner": min_straights_after_corner,
        "min_diags_after_diag": min_diags_after_diag,
        "quarters_random": quarters_random, "quarters_want": quarters_want,
        "semis_random": semis_random, "semis_want": semis_want,
        "holes_random": holes_random,
        "holes_circles": holes_circles,
        "holes_triangles": holes_triangles,
        "holes_quads": holes_quads,
        "fast_mode": fast_mode, "attempts": attempts, "correction_max_offset": correction_max_offset,
    })

    # Resolver cantidades de arcos para el generador
    quarters_arg = None if use_portable else (None if quarters_random else int(quarters_want))
    semis_arg    = None if use_portable else (None if semis_random else int(semis_want))

    # Huecos: si hay PS1, el panel se ignora; si está en aleatorio, dejamos None
    holes_spec = None if (use_portable or holes_random) else {
        "circles": int(holes_circles),
        "triangles": int(holes_triangles),
        "quads": int(holes_quads),
    }

    with st.spinner("Generando contorno (puede tardar si se buscan seeds cercanas)..."):
        try:
            ex = generate_contour(
                seed=None,  # sin seed base36: si no hay PS1, el core usa una aleatoria interna
                max_corners=max_corners,
                nx_range=(nx_min, nx_max),
                ny_range=(ny_min, ny_max),

                allow_diagonals_45=(max_diagonals > 0),
                max_diagonals=max_diagonals,

                min_straights_after_corner=min_straights_after_corner,
                min_diags_after_diag=min_diags_after_diag,
                lock_diag_opposite=True,  # sin control en UI; mantiene comportamiento seguro

                quarter_arcs_count=quarters_arg,
                semi_arcs_count=semis_arg,
                holes_spec=holes_spec,

                portable_seed=portable_seed_str if use_portable else None,

                # performance
                attempts=int(attempts),
                correction_max_offset=int(correction_max_offset),
            )
        except TypeError:
            # Compatibilidad si tu core no soporta attempts/offset
            ex = generate_contour(
                seed=None,
                max_corners=max_corners,
                nx_range=(nx_min, nx_max),
                ny_range=(ny_min, ny_max),
                allow_diagonals_45=(max_diagonals > 0),
                max_diagonals=max_diagonals,
                min_straights_after_corner=min_straights_after_corner,
                min_diags_after_diag=min_diags_after_diag,
                lock_diag_opposite=True,
                quarter_arcs_count=quarters_arg,
                semi_arcs_count=semis_arg,
                portable_seed=portable_seed_str if use_portable else None,
            )
        st.session_state["exercise"] = ex

if manual_btn:
    st.session_state["view"] = "manual"

# ───────────────────────────── Vista principal ─────────────────────────────
if st.session_state.get("view") == "manual":
    try:
        with open("docs/manual_completo.html", "r", encoding="utf-8") as f:
            html = f.read()
        # Eliminar enlaces internos del índice si causan problemas (preservar texto)
        import re
        html = re.sub(r"<a\s+href=\"(?:[^#\"]*#)?[^\"]*\">(.*?)</a>", r"\1", html, flags=re.IGNORECASE|re.DOTALL)
    except Exception as e:
        html = f"<div style='font-family:system-ui'><h2>Error al cargar el manual</h2><p>{e}</p></div>"
    components.html(html, height=900, scrolling=True)
    st.caption("<div style='text-align: center;'>Generador de figuras para calculo de centroide — Física I - Alfarano Javier</div>", unsafe_allow_html=True)
    st.stop()

ex = st.session_state.get("exercise")
st.markdown("### Ejercicio generado")
if ex:
    left, right = st.columns([2, 1], gap="large")

    with left:
        # Controles de visualización en 3 columnas centradas
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            show_grid = st.checkbox("Grilla debajo figura", value=st.session_state["show_grid"], key="show_grid_cb")
        with col2:
            show_grid_top = st.checkbox("Grilla sobre figura", value=st.session_state["show_grid_top"], key="show_grid_top_cb")
        with col3:
            show_centroid = st.checkbox("Mostrar centroide", value=st.session_state["show_centroid"], key="show_centroid_cb")

        # Colores separados de la generación del ejercicio
        st.subheader("Colores")
        col_color1, col_color2, col_color3, col_color4, col_color5 = st.columns(5)
        with col_color1:
            fill_color = st.color_picker("Relleno", st.session_state["fill_color"], key="fill_color_main")
            st.session_state["fill_color"] = fill_color
        with col_color2:
            stroke_color = st.color_picker("Borde", st.session_state["stroke_color"], key="stroke_color_main")
            st.session_state["stroke_color"] = stroke_color
        with col_color3:
            axis_color = st.color_picker("Ejes", st.session_state["axis_color"], key="axis_color_main")
            st.session_state["axis_color"] = axis_color
        with col_color4:
            grid_color = st.color_picker("Grilla debajo", st.session_state.get("grid_color", "#0a66c2"), key="grid_color_main")
            st.session_state["grid_color"] = grid_color
        with col_color5:
            grid_top_color = st.color_picker("Grilla sobre figura", st.session_state.get("grid_top_color", "#ffffff"), key="grid_top_color_main")
            st.session_state["grid_top_color"] = grid_top_color

        # Renderizar SVG
        svg = exercise_to_svg(
            ex,
            margin=30.0,
            show_centroid=show_centroid,
            fill_color=fill_color,
            stroke_color=stroke_color,
            axis_color=axis_color,
            grid_color=grid_color,
            show_grid=show_grid,
            show_grid_top=show_grid_top,
            grid_top_color=grid_top_color,
        )
        st.markdown(svg, unsafe_allow_html=True)

        # Botones de descarga debajo del gráfico en 2 columnas
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Descargar SVG",
                data=svg.encode("utf-8"),
                file_name=f"figura_{ex.get('seed','')}.svg",
                mime="image/svg+xml",
                use_container_width=True,
            )
        with col_dl2:
            png = exercise_to_png_bytes(
                ex,
                margin=30.0,
                show_centroid=show_centroid,
                scale=9,
                fill_color=fill_color,
                stroke_color=stroke_color,
                axis_color=axis_color,
                grid_color=grid_color,
                show_grid=show_grid,
                show_grid_top=show_grid_top,
                grid_top_color=grid_top_color,
            )
            st.download_button(
                "Descargar PNG",
                data=png,
                file_name=f"figura_{ex.get('seed','')}.png",
                mime="image/png",
                use_container_width=True,
            )

    with right:
        # ── Metadatos ──
        st.subheader("Metadatos")
        
        # Metadatos principales en una sola columna
        st.write(f"**Seed:** `{ex.get('portable_seed', '')}`")
        st.write(f"**Generador:** `{ex.get('generator_version', '')}`")
        st.write(f"**Grilla:** {ex.get('grid', {}).get('width', 0)}×{ex.get('grid', {}).get('height', 0)} celdas")
        
        if ex.get("portable_mode", False):
            st.info("Usando **seed portátil** → parámetros del panel **ignorados**.")

        # Centroide: siempre visible en el panel derecho
        ux = ex.get("solution_units", {}).get("ux", 0.0)
        uy = ex.get("solution_units", {}).get("uy", 0.0)
        st.write(f"**Centroide (unidades de grilla):** x = {ux:.2f}, y = {uy:.2f}")

        # Diagnóstico avanzado (campos de debug menos esenciales)
        corners = ex.get("debug", {}).get("corners", None)
        thr = ex.get("debug", {}).get("min_corners_threshold", None)
        rej = ex.get("debug", {}).get("reject_stats", {})
        status = ex.get("debug", {}).get("status", "ok")
        warnings = ex.get("debug", {}).get("warnings", [])
        cf = ex.get("circular_features", [])
        q = next((c for c in cf if c.get("type") == "quarter_arc"), {"count": 0, "placed": 0})
        s = next((c for c in cf if c.get("type") == "semi_arc"), {"count": 0, "placed": 0})
        tries = ex.get('debug', {}).get('tries_used', 0)
        fallback = ex.get('debug', {}).get('emergency_fallback', False)
        
        if any([corners is not None, thr is not None, rej, status != "ok", warnings, q.get('count', 0) > 0, s.get('count', 0) > 0, tries > 0]):
            with st.expander("Diagnóstico avanzado"):
                if corners is not None and thr is not None:
                    st.caption(f"**Complejidad:** esquinas = {corners} (umbral mínimo sin arcos = {thr})")
                if rej:
                    st.write("**Estadísticas de rechazo por intento:**")
                    st.json(rej)
                if status != "ok":
                    st.write(f"**Estado:** `{status}`")
                if warnings:
                    st.warning("Advertencias: " + ", ".join(warnings))
                # Metadatos técnicos movidos aquí
                st.write(f"**Seed (usuario):** `{ex.get('seed', '')}`")
                st.write(f"**Seed efectiva:** `{ex.get('seed_effective', '')}`")
                st.write(f"**Firma config:** `{ex.get('config_sig', '')}`")
                # Datos específicos movidos aquí
                st.write(f"**Cuartos de círculo —** solicitados: {q.get('count', 0)} · colocados: {q.get('placed', 0)}")
                st.write(f"**Semicírculos —** solicitados: {s.get('count', 0)} · colocados: {s.get('placed', 0)}")
                st.write(f"**Reintentos usados:** {tries}" + (" · **Fallback de emergencia**" if fallback else ""))

else:
    st.info("Generá un ejercicio desde el panel izquierdo.")

st.caption("<div style='text-align: center;'>Generador de figuras para calculo de centroide — Física I - Alfarano Javier</div>", unsafe_allow_html=True)
