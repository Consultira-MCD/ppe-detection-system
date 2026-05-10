# Argus Vision — PPE Detection System Dashboard
# Imagen mínima Python 3.11 (Debian Slim, sin GUI)
FROM python:3.11-slim

# Evitar prompts interactivos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema requeridas por OpenCV headless y ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar declaración de dependencias primero (aprovecha capa de caché de Docker)
COPY pyproject.toml .

# Instalar dependencias Python via pip
# opencv-python requiere libGL; en entornos headless (servidor) se puede cambiar
# a opencv-python-headless si no se necesita acceso a cámara desde el contenedor
RUN pip install --no-cache-dir \
        "altair<5" \
        "lapx>=0.9.4" \
        "matplotlib>=3.10.8" \
        "numpy>=2.4.2" \
        "onnxruntime>=1.24.4" \
        "opencv-python>=4.13.0.92" \
        "pandas>=3.0.1" \
        "seaborn>=0.13.2" \
        "streamlit>=1.19.0" \
        "ultralytics>=8.4.15"

# Copiar el código fuente y modelos
COPY src/ src/
COPY models/ models/
COPY main.py .

# Crear directorio de evidencias (logs CSV e imágenes en runtime)
RUN mkdir -p evidencias

# Puerto estándar de Streamlit
EXPOSE 8501

# Configuración de Streamlit: desactivar telemetría y modo headless
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "-m", "streamlit", "run", "src/04_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
