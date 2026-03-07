# PPE Detection System - Industrial Safety

> **Estado Actual del Proyecto:** 🚧 **Fase 4 en progreso (Dashboard y Visualización)**

Bienvenido al repositorio oficial del **Sistema de Detección de Equipo de Protección Personal (EPP)**. Este proyecto es una iniciativa de consultoría aplicada a la automatización de la seguridad industrial, utilizando visión artificial para la mitigación proactiva de riesgos laborales.

## Descripción del Proyecto

El objetivo principal es desarrollar y desplegar un sistema unificado basado en la arquitectura **YOLOv8** para la detección automatizada de EPP. El sistema evalúa el cumplimiento en tiempo real, con una estrategia de integración por etapas:

**Implementación Actual (v1.0):**

- 👷 Personas (Trabajadores)
- 🟡 Protección de Cabeza: Cascos de seguridad (`helmet` / `no-helmet`)
- 🦺 Protección de Torso: Chalecos de alta visibilidad (`vest` / `no-vest`)

**Expansión Planificada (v2.0):**

- 🧤 Protección de Manos: Guantes de seguridad
- 🥽 Protección Facial: Lentes/Goggles y Mascarillas

El sistema separa la detección visual de la lógica de negocio, permitiendo configurar qué implementos son obligatorios u opcionales dependiendo de la zona y normativas (ej. NOM-017 de la STPS).

## Estructura del Repositorio

- `data/raw/`: Imágenes originales sin procesar y archivos de anotación.
- `data/processed/`: Dataset optimizado y con Data Augmentation para evitar sesgos de color.
- `notebooks/`: Entorno de experimentación para el entrenamiento y Análisis Exploratorio de Datos (EDA).
- `src/`: Código fuente de producción y scripts de inferencia en tiempo real.
- `models/`: Pesos entrenados y versiones del modelo (ej. `v1_base.pt`, `v2_full.pt`).
- `reports/`: Gráficas de rendimiento (curvas PR, matrices de confusión, mAP).
- `docs/`: Documentación técnica, manuales y presentaciones comerciales.

## Hoja de Ruta (Roadmap) del Producto

### Fase 1: Prueba de Concepto (Motor Base v1.0)

- [x] Adquisición y formateo del dataset inicial (Roboflow).
- [x] Entrenamiento del modelo base mediante Transfer Learning (YOLOv8n).
- [x] Inferencia exitosa en tiempo real mediante webcam local.

### Fase 2: Expansión de Datos y Stress Test

- [x] **Mitigación de sesgo de color:** Implementar Data Augmentation (Hue/Saturation) para detectar chalecos de cualquier color (azul, rosa, negro), forzando al modelo a aprender patrones y no solo colores de alta visibilidad.
- [x] **Ampliación de clases (v2.0):** Integrar imágenes etiquetadas con guantes, lentes y mascarillas al dataset.
- [x] Pruebas de inferencia sobre videos pregrabados de obras y minas reales.

### Fase 3: Lógica de Negocio y Alertas

- [x] Implementación de tracking (seguimiento de IDs únicos por trabajador).
- [x] Programación de reglas lógicas de infracción (ej. "Alerta si ID_4 está sin casco por > 3 segundos").
- [x] Generación automática de logs e incidencias.

### Fase 4: Dashboard y Visualización (Cliente Final)

- [x] Desarrollo de interfaz gráfica web (Streamlit).
- [x] Integración de métricas en vivo (Trabajadores seguros vs. En riesgo).
- [x] Selector de fuentes de video.

### Fase 5: Despliegue en el Borde (Edge Computing)

- [ ] Contenerización del entorno (Docker).
- [ ] Optimización del modelo para hardware de bajo costo.

## Instalación y Configuración

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/Consultira-MCD/ppe-detection-system.git](https://github.com/Consultira-MCD/ppe-detection-system.git)
   ```

---
