import cv2
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURACIÓN PRINCIPAL (Fácil de editar)
# ==========================================
# Ruta exacta basada en la estructura de tu captura de pantalla
MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt" 
WEBCAM_INDEX = 0           # 0 para la cámara de la Mac, 1 para una externa
CONFIDENCE_THRESHOLD = 0.6 # Exigimos 60% de seguridad para evitar cuadros falsos

def get_optimal_device():
    """Detecta el mejor hardware disponible (Mac M4, NVIDIA o CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def main():
    print("Iniciando sistema de detección...")
    
    # 1. Configurar Hardware
    device = get_optimal_device()
    print(f"Hardware detectado: {device.upper()}")
    
    # 2. Cargar el modelo V2
    print(f"Cargando inteligencia desde: {MODEL_PATH}")
    modelo = YOLO(MODEL_PATH)
    
    # 3. Iniciar la cámara
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("\n¡Cámara activa! Presiona la tecla 'q' en la ventana de video para salir.")
    
    # 4. Bucle principal de inferencia
    while True:
        exito, frame = cap.read()
        if not exito:
            print("Se perdió la señal de la cámara.")
            break
            
        # Pasar la foto de la cámara al modelo
        resultados = modelo(
            frame, 
            conf=CONFIDENCE_THRESHOLD, 
            device=device, 
            verbose=False # Apagamos el texto en la terminal para que no se sature
        )
        
        # YOLO dibuja las cajas de colores automáticamente sobre el frame
        frame_anotado = resultados[0].plot()
        
        # Mostrar el resultado en una ventana
        cv2.imshow("Sistema de Deteccion EPP - V2", frame_anotado)
        
        # Esperar 1 milisegundo a ver si el usuario presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando sistema...")
            break
            
    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()