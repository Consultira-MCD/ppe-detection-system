<table width="100%" style="border:none;border-collapse:collapse;" cellspacing="0" cellpadding="16">
  <tr>
    <td width="20%" align="center" valign="middle" style="border:none;">
      <img src="miscellaneous/mcd.png" alt="MCD Logo" width="150"/>
    </td>
    <td width="80%" valign="middle" style="border:none;">
      <h1>PPE Detection System — Argus Vision</h1>
      <p>Sistema de detección de Equipo de Protección Personal en tiempo real mediante visión computacional.</p>
      <p>
        <strong>Autores:</strong> Francisco Ortega &amp; Jose Cazares<br/>
        <strong>Programa:</strong> Maestría en Ciencia de Datos (MCD)<br/>
        <strong>Organización:</strong> Sahuaro Data Analytics
      </p>
    </td>
  </tr>
</table>

---

> **Estado Actual:** Fase 4 completada. Modelo `yolov8_epp_v2` en producción, integrado en el Dashboard Argus Vision.

Repositorio oficial del **Sistema de Detección de Equipo de Protección Personal (EPP) — Argus Vision**, una iniciativa de consultoría aplicada a la automatización de la seguridad industrial. El sistema utiliza visión computacional para la evaluación proactiva y en tiempo real del cumplimiento de normativas de seguridad laboral (NOM-017, STPS).

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Evolución del Repositorio por Rama](#evolución-del-repositorio-por-rama)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Hoja de Ruta (Roadmap)](#hoja-de-ruta-roadmap)
- [Trabajos Futuros](#trabajos-futuros)
- [Instalación y Configuración](#instalación-y-configuración)

---

## Descripción del Proyecto

El objetivo principal es desarrollar y desplegar un sistema unificado basado en la arquitectura **YOLOv8** para la detección automatizada de EPP. El sistema evalúa el cumplimiento en tiempo real, con una estrategia de integración y entrenamiento por etapas.

**Clases detectadas en producción (v2.0 — `yolov8_epp_v2`):**

| Clase | Descripción |
|---|---|
| `head_helmet` / `head_nohelmet` | Casco de seguridad |
| `vest` | Chaleco de alta visibilidad |
| `glasses` | Lentes de seguridad |
| `face_mask` / `face_nomask` | Mascarilla de protección |
| `person` | Trabajador detectado |

**Clases del modelo base (v1.0 — Prueba de Concepto):**

- `helmet` / `no-helmet`, `vest` / `no-vest`, `person`

---

## Evolución del Repositorio por Rama

El historial del proyecto está estructurado en ramas independientes, cada una representando una fase completa de desarrollo. La rama `main` siempre contiene el estado más reciente y estable del sistema.

```
main          ← Estado más reciente: modelo yolov8_epp_v2 + Dashboard Argus Vision
│
├── v4_fase4  ← Experimentación con pipeline progresivo en La Yuca (AMD MI210)
├── v3_fase3  ← Lógica de negocio, tracking multi-persona y generación de alertas
├── v2_fase2  ← Dataset expandido (~30k imágenes), entrenamiento en GPU A100 ← modelo en uso
└── v1_fase1  ← Prueba de concepto: modelo base, inferencia local en webcam
```

### `v1_fase1` — Prueba de Concepto
Primera iteración funcional del sistema. Se entrenó un modelo `YOLOv8n` con Transfer Learning sobre un dataset de construcción descargado desde Roboflow (~800 imágenes). El entrenamiento se realizó en local con **Apple Silicon (MPS)** durante 25 épocas a 640px. Se logró inferencia exitosa en webcam con las clases base: `helmet`, `no-helmet`, `vest`, `no-vest`, `person`.

> Libreta: `notebooks/01_entrenamiento_epp.ipynb`

### `v2_fase2` — Robustez Industrial y Eliminación de Sesgo
Se identificó que el modelo v1 dependía del color amarillo/naranja de los chalecos. En esta fase se mitigó el sesgo implementando **Data Augmentation agresivo** (Hue ±15°, escala de grises 15%, Blur 2px, Flip horizontal), escalando el dataset a casi **30,000 imágenes** (11 clases). El entrenamiento migró a la nube con una **GPU NVIDIA A100-SXM4-80GB** en Google Colab: 100 épocas, batch 32, Early Stopping con paciencia de 30.

> Libreta: `notebooks/02_entrenamiento_epppv2.ipynb`

### `v3_fase3` — Lógica de Negocio y Alertas
Integración de la capa de inteligencia operativa sobre la detección visual: **tracking de identidades únicas** por trabajador (BotSort), **reglas de infracción configurables** (e.g., "alerta si ID_4 está sin casco por > 3 segundos"), **generación automática de logs CSV** y capturas de evidencia fotográfica con timestamp. Se construyó la UI personalizada en OpenCV (`src/03_custom_ui.py`).

> Script: `src/03_custom_ui.py`

### `v4_fase4` — Experimentación con Pipeline Progresivo (La Yuca)
Fase de investigación y experimentación avanzada. Se diseñó un pipeline de **Progressive Resizing** en 4 sub-fases encadenadas sobre el dataset **SH17** (~8,000 imágenes industriales, 17 clases), utilizando el **Centro de Supercómputo "La Yuca"** con **GPU AMD Instinct MI210** (ROCm). Los resultados de esta experimentación informaron decisiones de diseño del sistema final, aunque el modelo desplegado en producción es `yolov8_epp_v2`.

| Sub-fase | Resolución | Objetivo |
|---|---|---|
| 1/4 — Fundación | 640px | Aprender formas y clases base |
| 2/4 — Refinamiento | 1280px | Detección de EPP de menor tamaño |
| 3/4 — Alta Resolución | 1920px | Máxima precisión con augmentation extremo |
| 4/4 — Fine-Tuning Final | 1920px | Estabilización (lr=0.0001, AdamW, cosine schedule) |

> Libreta: `notebooks/03_entrenamiento_yuca.ipynb` | Pesos experimentales: `models/fase4_modelo_final_produccion/weights/`

### `main` — Estado Actual
Integra todos los avances previos. El **modelo `yolov8_epp_v2`** (entrenado con ~30,000 imágenes y 11 clases) es el que corre en producción. Incluye el **Dashboard Operativo Argus Vision** (`src/04_dashboard.py`), una interfaz web construida con **Streamlit** que muestra métricas en tiempo real, soporte multi-cámara, registro de infracciones con evidencia y exportación de reportes HSE.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     ARGUS VISION — HSE Monitor               │
├─────────────────────────────────────────────────────────────┤
│  Fuentes de Entrada         │  Motor de Detección            │
│  ─────────────────          │  ───────────────────           │
│  • Webcam local             │  YOLOv8n (yolov8_epp_v2)       │
│  • Video pregrabado (MP4)   │  BotSort Tracker               │
│  • Stream RTSP (IP Cam)     │  conf=0.6, persist tracking    │
├─────────────────────────────────────────────────────────────┤
│  Lógica de Negocio          │  Salidas                       │
│  ──────────────             │  ──────                        │
│  • Reglas de infracción     │  • Dashboard Streamlit         │
│  • Timer de confirmación    │  • Logs CSV (One-Hot Encoding) │
│  • Cooldown anti-spam 30s   │  • Evidencia fotográfica       │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline de Entrenamiento

El proyecto documenta tres generaciones de entrenamiento, cada una con su libreta correspondiente:

```
notebooks/
├── 01_entrenamiento_epp.ipynb       ← Fase 1: TL local, 25 epochs, 640px (MPS)
├── 02_entrenamiento_epppv2.ipynb    ← Fase 2: A100, 100 epochs, 30k imgs, 11 clases  ← PRODUCCIÓN
└── 03_entrenamiento_yuca.ipynb      ← Fase 4: AMD MI210, SH17, Progressive Resizing (experimental)
```

**Hiperparámetros del modelo en producción (`yolov8_epp_v2`):**

| Parámetro | Valor |
|---|---|
| Arquitectura | YOLOv8n (Transfer Learning) |
| Dataset | ~30,000 imágenes (11 clases de EPP) |
| Resolución | 640 × 640 px |
| Épocas | 100 (Early Stopping, paciencia 30) |
| Batch Size | 32 |
| Hardware | GPU NVIDIA A100-SXM4-80GB (Google Colab) |
| Data Augmentation | Hue ±15°, Grayscale 15%, Blur 2px, Flip horizontal |
| Exportación | PyTorch `.pt` |

---

## Estructura del Repositorio

```
ppe-detection-system/
├── notebooks/              # Libretas de experimentación y entrenamiento documentadas
│   ├── 01_entrenamiento_epp.ipynb      # Fase 1: Prueba de concepto
│   ├── 02_entrenamiento_epppv2.ipynb   # Fase 2: Robustez y dataset expandido
│   └── 03_entrenamiento_yuca.ipynb     # Fase 4: Producción en supercomputadora
│
├── src/                    # Código de producción e inferencia
│   ├── 00_check_test.py    # Validación de entorno y hardware
│   ├── 01_model_test_V1.py # Pruebas de inferencia modelo v1
│   ├── 02_model_test_V2.py # Pruebas de inferencia modelo v2
│   ├── 03_custom_ui.py     # UI personalizada en OpenCV + lógica de alertas
│   └── 04_dashboard.py     # Dashboard operativo Argus Vision (Streamlit)
│
├── models/                 # Pesos entrenados por versión
│   ├── yolov8_epp_v1/                      # Modelo Fase 1
│   ├── yolov8_epp_v2/                      # Modelo Fase 2
│   └── fase4_modelo_final_produccion/      # Modelo de producción
│       └── weights/
│           ├── best.pt     # Mejores pesos PyTorch
│           └── best.onnx   # Exportación ONNX (portabilidad)
│
├── data/
│   ├── raw/                # Datasets originales (Roboflow, Kaggle/SH17)
│   └── processed/          # Dataset con Data Augmentation aplicado
│
└── reports/                # Gráficas de rendimiento (PR curves, mAP, confusión)
```

---

## Hoja de Ruta (Roadmap)

### Fase 1: Prueba de Concepto — Motor Base v1.0 ✅
- [x] Adquisición del dataset inicial desde Roboflow (~800 imágenes).
- [x] Transfer Learning con `YOLOv8n` en hardware local (Apple Silicon MPS).
- [x] Inferencia en tiempo real vía webcam (25 epochs, 640px).
- [x] Detección funcional: `helmet`, `no-helmet`, `vest`, `no-vest`, `person`.

### Fase 2: Expansión de Datos y Robustez Industrial ✅
- [x] Mitigación de sesgo de color con Data Augmentation (Hue/Grayscale/Blur/Flip).
- [x] Escalamiento del dataset a ~30,000 imágenes (11 clases de EPP).
- [x] Entrenamiento en la nube: GPU NVIDIA A100-SXM4-80GB (Google Colab).
- [x] Pruebas de inferencia en videos pregrabados de obras y minas reales.

### Fase 3: Lógica de Negocio y Sistema de Alertas ✅
- [x] Tracking de identidades únicas por trabajador (BotSort).
- [x] Reglas de infracción configurables con timer de confirmación.
- [x] Generación automática de logs CSV y evidencia fotográfica con timestamp.
- [x] UI personalizada en OpenCV con cooldown anti-spam.

### Fase 4: Dashboard HSE y Sistema Completo ✅
- [x] Dashboard operativo web con Streamlit (métricas en vivo, soporte multi-cámara).
- [x] Integración del modelo `yolov8_epp_v2` como motor de detección en producción.
- [x] Experimentación con pipeline de Progressive Resizing en supercomputadora La Yuca.
- [x] Sistema completo funcional: detección → tracking → alertas → evidencia → reporte.

---

## Trabajos Futuros

Las siguientes iniciativas representan líneas de desarrollo identificadas para extender el alcance del sistema:

- **Despliegue en el Borde (Edge Computing):** Contenerización con Docker y optimización del modelo para hardware embebido de bajo costo (Jetson Nano, Raspberry Pi) mediante cuantización INT8.
- **Ampliación de cobertura de EPP:** Incorporar al modelo en producción las clases adicionales entrenadas en la Fase 4 (lentes, guantes, protección auditiva) una vez validadas en campo.
- **Integración con sistemas CCTV existentes:** Adaptadores para protocolos RTSP/ONVIF para conectar el sistema directamente con la infraestructura de cámaras industriales.
- **API REST de reportes:** Exponer las métricas e incidentes mediante una API para integración con plataformas ERP/HSE corporativas.
- **Modelo multi-zona:** Soporte para reglas de infracción diferenciadas por área (p. ej., zona de soldadura vs. zona de oficinas) con configuración por cámara.

---

## Instalación y Configuración

### Requisitos
- Python 3.11+
- `uv` (gestor de entornos recomendado) o `pip`

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Consultira-MCD/ppe-detection-system.git
cd ppe-detection-system

# 2. Crear el entorno virtual e instalar dependencias
uv sync
# o con pip:
pip install -r requirements.txt

# 3. Ejecutar el dashboard operativo
uv run streamlit run src/04_dashboard.py

# 4. Ejecutar inferencia con UI personalizada (OpenCV)
uv run python src/03_custom_ui.py
```

### Explorar una fase específica

```bash
# Cambiar a una rama para ver el estado del proyecto en esa fase
git checkout v1_fase1   # Prueba de concepto
git checkout v2_fase2   # Dataset expandido
git checkout v3_fase3   # Lógica de negocio y alertas
git checkout v4_fase4   # Modelo final + pipeline La Yuca
git checkout main       # Estado más reciente
```

---

*Proyecto desarrollado por Francisco Ortega — Sahuaro Data Analytics*
