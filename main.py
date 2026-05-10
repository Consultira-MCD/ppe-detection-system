"""
PPE Detection System — Argus Vision
Sistema de detección de Equipo de Protección Personal en tiempo real.

Comandos de uso:
  uv run streamlit run src/04_dashboard.py   → Dashboard operativo HSE (recomendado)
  uv run python src/03_custom_ui.py          → Interfaz OpenCV con lógica de alertas
  uv run python src/00_check_test.py         → Validar entorno y hardware disponible
"""


def main():
    print("=" * 60)
    print("  ARGUS VISION — PPE Detection System")
    print("=" * 60)
    print()
    print("Para iniciar el Dashboard HSE ejecuta:")
    print("  uv run streamlit run src/04_dashboard.py")
    print()
    print("Para la interfaz OpenCV (terminal):")
    print("  uv run python src/03_custom_ui.py")
    print()
    print("Para validar el entorno:")
    print("  uv run python src/00_check_test.py")
    print()
    print("Modelo en producción: models/yolov8_epp_v2_produccion/weights/best.pt")
    print("  mAP50: 0.907 | mAP50-95: 0.613 | Clases: 11 | Épocas: 100")


if __name__ == "__main__":
    main()
