# Generador de Figuras para Cálculo de Centroide

**Versión:** 2.6.2  
**Autor:** Alfarano Javier  
**Asignatura:** Física I  

---

## Cómo ejecutar la aplicación

### Requisitos previos
- Python 3.9 o superior  
- Git (opcional, para clonar el repositorio)  

### Instalación
Clonar el repositorio o descargarlo en tu máquina:
```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
```

Crear entorno virtual e instalar dependencias:
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Ejecución local
```bash
streamlit run ui/streamlit_app.py
```

La aplicación estará disponible en [http://localhost:8501](http://localhost:8501).

---

## Descripción

El **Generador de Figuras para Cálculo de Centroide** es una herramienta educativa desarrollada en **Python** con **Streamlit**, orientada a estudiantes de ingeniería y física.  

Permite crear **ejercicios reproducibles** de cálculo de centroides con figuras geométricas que combinan contornos ortogonales, diagonales, arcos circulares y huecos internos.  

El sistema calcula automáticamente el centroide y permite **exportar las figuras en formatos SVG y PNG**, ofreciendo un recurso práctico para clases, prácticas y evaluaciones.

---

## Funcionalidades

- **Generación procedural de figuras:** cada ejercicio se crea a partir de una *seed* que garantiza reproducibilidad.  
- **Configuración flexible:** número de esquinas, diagonales, arcos, huecos y dimensiones de la grilla.  
- **Cálculo automático del centroide:** coordenadas exactas en píxeles y en unidades de grilla.  
- **Exportación:** descarga en **SVG** y **PNG**.  
- **Interfaz intuitiva:** controles en panel lateral y vista dinámica en navegador.  
- **Manual completo:** disponible en [`docs/manual_completo.html`](docs/manual_completo.html).  

---

## Estructura del proyecto

```
repo/
├── core/                 # Lógica principal de generación de figuras
│   ├── generator_contour.py
│   ├── contour.py
│   ├── seed.py
│   └── __init__.py
├── ui/                   # Interfaz de usuario y renderizado
│   ├── streamlit_app.py  # Punto de entrada (UI con Streamlit)
│   ├── render_svg.py
│   └── render_png.py
├── docs/                 # Documentación y manual completo
│   └── manual_completo.html
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este archivo
```

---

## Manual

El manual completo con fundamentos matemáticos, algoritmos, parámetros configurables, ejemplos de uso y solución de problemas se encuentra en:  

👉 [`docs/manual_completo.html`](docs/manual_completo.html)  

---

## Propósito

El objetivo es ofrecer una **herramienta educativa práctica y visual**, que facilite el aprendizaje del cálculo de centroides y promueva la experimentación con problemas similares a los que enfrentan ingenieros y físicos en contextos reales.

---

# Figures Generator for Centroid Calculation

**Version:** 2.6.2  
**Author:** Alfarano Javier  
**Subject:** Physics I  

---

## How to run the application

### Requirements
- Python 3.9 or higher  
- Git (optional, to clone the repository)  

### Installation
Clone the repository or download it:
```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Run locally
```bash
streamlit run ui/streamlit_app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

---

## Description

The **Figures Generator for Centroid Calculation** is an educational tool built in **Python** with **Streamlit**, designed for engineering and physics students.  

It allows creating **reproducible exercises** for centroid calculation using geometric figures that combine orthogonal contours, diagonals, circular arcs, and inner holes.  

The system automatically calculates the centroid and allows exporting figures in **SVG** and **PNG** formats, providing a practical resource for classes, exercises, and evaluations.

---

## Features

- **Procedural figure generation:** each exercise is based on a *seed* for reproducibility.  
- **Flexible configuration:** number of corners, diagonals, arcs, holes, and grid dimensions.  
- **Automatic centroid calculation:** precise coordinates in pixels and grid units.  
- **Export options:** download in **SVG** and **PNG**.  
- **Interactive interface:** sidebar controls and dynamic visualization.  
- **Complete manual:** available at [`docs/manual_completo.html`](docs/manual_completo.html).  

---

## Project structure

```
repo/
├── core/                 # Core logic for figure generation
│   ├── generator_contour.py
│   ├── contour.py
│   ├── seed.py
│   └── __init__.py
├── ui/                   # User interface and rendering
│   ├── streamlit_app.py  # Entry point (Streamlit UI)
│   ├── render_svg.py
│   └── render_png.py
├── docs/                 # Documentation and full manual
│   └── manual_completo.html
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## Manual

The complete manual with mathematical foundations, algorithms, configurable parameters, usage examples, and troubleshooting is available at:  

👉 [`docs/manual_completo.html`](docs/manual_completo.html)  

---

## Purpose

The goal is to provide a **practical and visual educational tool**, making centroid calculation easier to learn while encouraging experimentation with problems similar to those faced by engineers and physicists in real-world contexts.
