# PPE Detection System - Industrial Safety

Bienvenido al repositorio oficial del **Sistema de Detección de Equipo de Protección Personal (EPP)**. Este proyecto es una iniciativa de consultoría aplicada a la seguridad industrial, utilizando visión artificial avanzada para la mitigación de riesgos laborales.

## Descripción del Proyecto
El objetivo principal es desarrollar y desplegar un modelo de Deep Learning basado en la arquitectura **YOLO (You Only Look Once)** para la detección automatizada de EPP en entornos industriales. El sistema está diseñado para identificar:
* **Cascos de seguridad**
* **Chalecos de alta visibilidad**
* **Lentes de protección / Goggles**

Este desarrollo busca reducir los tiempos de supervisión manual y aumentar el cumplimiento de las normativas de seguridad (como la NOM en México o estándares OSHA) mediante el monitoreo inteligente de imágenes y video en tiempo real.

---

## Estructura del Proyecto
Basado en el estándar de nuestra organización, el repositorio se organiza de la siguiente manera:

* **`data/`**: 
    * `raw/`: Imágenes originales sin procesar y archivos de anotación.
    * `processed/`: Dataset optimizado, redimensionado y aumentado listo para entrenamiento.
* **`notebooks/`**: Experimentos en Jupyter/Colab para el Análisis Exploratorio de Datos (EDA) y el entrenamiento del modelo.
* **`src/`**: Código fuente de la aplicación, incluyendo scripts de inferencia para webcam y lógica de procesamiento de video.
* **`models/`**: Pesos entrenados (`.pt` o `.onnx`) y archivos de configuración de hiperparámetros.
* **`reports/`**: Reportes de performance, matrices de confusión, curvas PR y métricas de precisión (mAP).
* **`docs/`**: Documentación técnica, manuales de usuario y referencias industriales.

---

## Hoja de Ruta (Roadmap)
- [ ] **Fase 1: Data Engineering** - Recopilación y etiquetado de imágenes (vía Roboflow/CVAT).
- [ ] **Fase 2: Model Training** - Entrenamiento mediante Transfer Learning con YOLO.
- [ ] **Fase 3: Prototipado de Inferencia** - Implementación de detección sobre imágenes estáticas.
- [ ] **Fase 4: Real-time Deployment** - Integración con feed de video/webcam mediante OpenCV.
- [ ] **Fase 5: Reporting** - Generación de logs de cumplimiento e incidencias detectadas.

---

## Instalación y Configuración
1. Clona este repositorio:
   ```bash
   git clone [https://github.com/Consultira-MCD/ppe-detection-system.git](https://github.com/Consultira-MCD/ppe-detection-system.git)

---

