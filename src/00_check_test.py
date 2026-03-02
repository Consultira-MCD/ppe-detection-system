"""
Script de validación de entorno multiplataforma.
Detecta automáticamente el hardware disponible 
(CUDA para Windows/Linux con NVIDIA, 
MPS para Mac Apple Silicon, o CPU como respaldo) y
ejecuta una prueba de inferencia con YOLO.
"""

import torch
import cv2
from ultralytics import YOLO

# Función para detectar si el hardware es Mac o Windows/Linux
def get_optimal_device():
    # Detecto el hardware de aceleración disponible según el sistema operativo.
    if torch.cuda.is_available():
        print("Hardware detectado: CUDA. Utilizando GPU NVIDIA (Windows/Linux).")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("Hardware detectado: MPS. Utilizando Apple Silicon GPU (Mac).")
        return 'mps'
    else:
        print("Advertencia: No se detectó aceleración por hardware. Utilizando CPU.")
        return 'cpu'

# Funcion Principal. Ejecuta el script. 
def main():
    # 1. Configuración del dispositivo
    device = get_optimal_device()

    # 2. Inicialización del modelo
    # Utilizo yolov8n.pt por ser la variante más ligera para pruebas en tiempo real.
    print("Cargando los pesos del modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # 3. Configuración de la entrada de video
    # cv2.VideoCapture(0) le dice a la computadora que abra la cámara número 0 
    # (la webcam por defecto). Si tuvieras una cámara USB externa, podría ser el índice 1 o 2.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se logró establecer conexión con la cámara web.")
        return

    print("Captura de video iniciada. Presionar 'q' para terminar el proceso.")

    # 4. Bucle principal de inferencia
    # Un video es solo una secuencia rápida de fotos (frames). 
    # Este bucle 'While True' captura una foto tras otra sin detenerse.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Pérdida de frames en la lectura de la cámara.")
            break
        
        # Ejecuto la inferencia sobre el frame actual usando el hardware detectado.
        # verbose=False evita saturar la terminal con los logs de cada predicción.
        results = model(frame, device=device, verbose=False)
        
        # Genero un nuevo frame con las cajas delimitadoras (bounding boxes) dibujadas.
        annotated_frame = results[0].plot()
        
        # Muestro el resultado en pantalla.
        cv2.imshow("Test de Inferencia YOLO - EPP", annotated_frame)
        
        # Condición para interrumpir el bucle.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando la captura de video...")
            break

    # 5. Liberación de memoria y recursos
    # Siempre debemos apagar la cámara y destruir las ventanas de video 
    # para no dejar procesos trabados en la memoria de la Mac.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()