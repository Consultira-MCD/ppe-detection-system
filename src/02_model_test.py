import cv2
from ultralytics import YOLO

def iniciar_camara():
    print("Cargando el modelo entrenado...")
    # Asegúrate de que esta ruta apunte a donde está tu best.pt
    # Como el script está en src/, subimos un nivel con '../'
    modelo = YOLO('runs/models/yolov8_epp_v1/weights/best.pt')

    print("Iniciando cámara web... Presiona la tecla 'q' en tu teclado para cerrar la ventana.")
    # El 0 indica la cámara web principal de la Mac
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        exito, frame = cap.read()
        
        if not exito:
            print("No se pudo acceder a la cámara.")
            break
            
        # Pasar el frame actual de la cámara por el modelo YOLO
        resultados = modelo(frame, conf=0.5, verbose=False)
        
        # Dibujar las cajas delimitadoras sobre el frame
        frame_anotado = resultados[0].plot()
        
        # Mostrar el video en vivo en una ventana emergente
        cv2.imshow("Deteccion de EPP - Tiempo Real", frame_anotado)
        
        # Romper el ciclo si el usuario presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar los recursos de la Mac al terminar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_camara()