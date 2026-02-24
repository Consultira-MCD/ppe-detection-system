#  [Nombre de la Consultoría] - Plantilla Base para Proyectos de Datos

Bienvenido al repositorio de plantilla oficial. Este *boilerplate* está diseñado para estandarizar la estructura de nuestros proyectos de datos y consultoría. 

Usar esta misma estructura en todos los proyectos nos permite a todo el equipo colaborar de forma más eficiente, entender rápidamente el trabajo de los demás y mantener las mismas buenas prácticas.

Si vez que hace falta agregar algo, favor de agregarlo.

---

## 📁 Estructura del Proyecto

Esta es la organización estándar de nuestras carpetas. **Por favor, mantén esta estructura al inicializar tu proyecto.**

```text
├── data/               #  NUNCA modificar los datos en esta carpeta
│   ├── raw/            # Datos originales, inmutables y tal cual se recibieron
│   └── processed/      # Datos limpios y transformados, listos para modelar
├── notebooks/          # Jupyter/Colab notebooks para exploración (EDA) y pruebas
├── src/                # Código fuente principal (scripts, funciones, clases)
├── models/             # Modelos entrenados y serializados (ej. .pkl, .h5)
├── reports/            # Reportes generados, presentaciones y análisis finales
│   └── figures/        # Gráficas e imágenes exportadas
├── docs/               # Documentación extra, minutas, manuales o referencias
├── .gitignore          # Archivos y carpetas que Git debe ignorar
├── requirements.txt    # Dependencias y librerías necesarias para ejecutar el proyecto
└── README.md           # Este documento
