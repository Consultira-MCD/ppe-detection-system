"""
Fase 1 del proyecto de la consultoría.

Script de prueba del modelo entrenado de detección de EPP.
Abre la cámara web y ejecuta el modelo YOLO con los pesos obtenidos (best.pt)
para detectar elementos de protección personal en tiempo real.
"""

# Importamos la librería OpenCV (cv2) para el manejo de la captura 
# de video y procesamiento de la imagen
import cv2

# Importamos la clase YOLO de la librería ultralytics para poder cargar 
# nuestro modelo entrenado
from ultralytics import YOLO

def iniciar_camara():
    """
    Función que inicializa el modelo pre-entrenado y comienza 
    el ciclo de lectura de la cámara web para realizar inferencias 
    en tiempo real sobre cada fotograma.
    """
    # Notificamos al usuario que el modelo está a punto de cargarse en memoria
    print("Cargando el modelo entrenado...")
    
    # Cargamos el modelo YOLOv8 utilizando los pesos ('weights') 
    # del mejor modelo guardado ('best.pt').
    # Este modelo ('best.pt') contiene nuestro entrenamiento específico 
    # de detección de EPP.
    modelo = YOLO('models/yolov8_epp_v1/weights/best.pt')

    # Damos la instrucción al usuario sobre cómo detener la ejecución 
    # de manera segura
    print("Iniciando cámara web... Presiona la tecla 'q' en tu teclado para cerrar la ventana.")
    
    # Inicializamos la captura de video conectándonos a la cámara web 
    # por defecto de la computadora (índice 0)
    cap = cv2.VideoCapture(0)

    # Entramos en un ciclo de ejecución continua mientras la conexión 
    # de la cámara se mantenga estable
    while cap.isOpened():
        # Capturamos el fotograma actual de la cámara. 'exito' 
        # es un booleano que avisa si se leyó bien, 'frame' es la imagen
        exito, frame = cap.read()
        
        # Validamos si hubo una interrupción en la señal o si no se 
        # concedieron permisos de cámara
        if not exito:
            # Informamos por terminal del problema de lectura para diagnóstico
            print("No se pudo acceder a la cámara o se interrumpió la lectura del video.")
            # Rompemos el ciclo continuo al no tener más fotogramas que procesar
            break
            
        # Ejecutamos la predicción (inferencia) pasando el fotograma 
        # recién capturado al modelo YOLO. 
        # conf=0.5: Filtramos las detecciones que tienen menos del 50% de confianza.
        # verbose=False: Silenciamos los abundantes logs técnicos de YOLO para mantener limpia nuestra consola.
        resultados = modelo(frame, conf=0.5, verbose=False)
        
        # Extraemos el primer (y único) set de resultados y pedimos a la librería generar la imagen con cuadros delimitadores
        frame_anotado = resultados[0].plot()
        
        # Generamos (o actualizamos) una ventana en el sistema operativo para mostrar nuestra imagen ya procesada ('frame_anotado')
        cv2.imshow("Deteccion de EPP - Tiempo Real", frame_anotado)
        
        # Interceptamos la entrada del teclado por 1 milisegundo en cada ciclo del bucle.
        # Extraemos los últimos 8 bits y validamos si concuerdan con el código ASCII de la letra 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Si el usuario presionó 'q', rompemos limpiamente el ciclo infinito
            break

    # Una vez terminado el ciclo, liberamos el hardware de la cámara para que quede disponible al resto del sistema operativo
    cap.release()
    
    # Cerramos sistemáticamente todas las ventanas emergentes generadas por OpenCV para liberar memoria visual
    cv2.destroyAllWindows()

# Punto de entrada estándar de los scripts de Python. Comprobamos si este archivo fue ejecutado explícitamente y no importado
if __name__ == "__main__":
    # De ser así, ejecutamos la función controladora de inicialización de la cámara e inferencias
    iniciar_camara()