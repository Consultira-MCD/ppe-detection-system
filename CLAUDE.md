# CLAUDE.md — Argus Vision PPE Detection System

Guía de contexto para Claude Code en este proyecto.

## Comandos esenciales

```bash
# Dashboard operativo HSE (entrada principal)
uv run streamlit run src/04_dashboard.py

# Inferencia con UI OpenCV y lógica de alertas
uv run python src/03_custom_ui.py

# Validar entorno y hardware disponible (MPS/CUDA/CPU)
uv run python src/00_check_test.py

# Instalar dependencias
uv sync
```

## Estructura del proyecto

```
src/
  00_check_test.py    # Validación de entorno y hardware
  01_model_test_V1.py # Pruebas modelo v1 (baseline)
  02_model_test_V2.py # Pruebas modelo v2
  03_custom_ui.py     # UI OpenCV con tracking BotSort y logs CSV
  04_dashboard.py     # Dashboard Streamlit (~800 líneas, producción)

models/
  yolov8_epp_v1/                        # Fase 1, baseline (mAP50=0.857)
  yolov8_epp_v2_produccion/weights/     # MODELO EN USO (mAP50=0.907)
    best.pt                             # Mejores pesos PyTorch
    last.pt
  yolov8_epp_v3/                        # Fase 4 experimental (mAP50=0.628)
    fase4_modelo_final_produccion/weights/
      best.pt
      best.onnx                         # Exportación ONNX

notebooks/
  01_entrenamiento_epp.ipynb            # Fase 1
  02_entrenamiento_epppv2.ipynb         # Fase 2 (producción)
  03_entrenamiento_yuca.ipynb           # Fase 4 (La Yuca, AMD MI210)
```

## Modelo en producción

- **Archivo:** `models/yolov8_epp_v2_produccion/weights/best.pt`
- **mAP50:** 0.907 | **mAP50-95:** 0.613
- **Clases (11):** `head_helmet`, `head_nohelmet`, `vest`, `glasses`, `face_mask`, `face_nomask`, `person`, y variantes
- **Hardware de entrenamiento:** NVIDIA A100-SXM4-80GB (Google Colab)
- **Dataset:** ~30,000 imágenes con Data Augmentation (Hue ±15°, Grayscale 15%, Blur 2px, Flip H)

## Schema CSV de incidencias

Ambos scripts (`03_custom_ui.py` y `04_dashboard.py`) escriben al mismo archivo `evidencias/reporte_incidencias.csv` con este schema:

```
Fecha, Hora, ID_Persona, Chaleco, Casco, Lentes, Mascarilla, Nombre_Foto
```

Los valores de EPP son 1 = falta el equipo, 0 = equipo presente.

## Restricciones conocidas

- **PyArrow / Streamlit:** Si Streamlit lanza error de compatibilidad con PyArrow al leer el CSV, el dashboard tiene un bloque `try/except` que recae en `use_arrow=False`. No tocar ese bloque.
- **Rutas de modelo relativas:** Todos los `MODEL_PATH` usan rutas relativas desde la raíz del proyecto. Ejecutar siempre desde la raíz, no desde `src/`.
- **`__MACOSX/`** está en `.gitignore`. Si aparece en git status tras descomprimir un zip en macOS, usar `git add -u` para limpiar.
- El directorio `evidencias/` está en `.gitignore` (contiene imágenes y CSV generados en runtime).

## Dependencias

Gestionadas con `uv` y declaradas en `pyproject.toml`. Python 3.11+ requerido. No existe `requirements.txt`; usar siempre `uv sync`.

## Convenciones

- Los notebooks son documentación de experimentación, no código de producción.
- El código de producción vive en `src/`. Los scripts se nombran con prefijo numérico indicando orden de uso.
- No hacer commit de pesos de modelos (`.pt`, `.onnx`) a menos que sean intencionados; son archivos grandes.
