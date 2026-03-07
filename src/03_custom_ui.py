"""
Script principal de la interfaz de usuario personalizada (Custom UI).
Ejecuta el modelo YOLO para detectar equipo de protección personal (EPP),
evalúa en tiempo real si el personal cumple con el equipo completo y 
genera evidencias (fotografías y registros CSV en formato One-Hot Encoding)
en caso de detectar una infracción validada por tiempo.
"""

# Importamos OpenCV para la captura de video y manipulación visual de la imagen (UI)
import cv2

# Importamos PyTorch para manejar operaciones tensoriales y consultar aceleración por hardware
import torch

# Importamos os para crear carpetas e interactuar con el sistema de archivos (gestión de evidencias)
import os

# Importamos time para llevar seguimiento de tiempos de enfriamiento y confirmación de faltas
import time

# Importamos csv para estructurar, escribir y almacenar los registros de incidencias detectadas
import csv

# Importamos datetime para estampar la fecha y hora exacta en las fotos y registros CSV
from datetime import datetime

# Importamos la clase principal de la librería de YOLO para cargar el modelo de IA
from ultralytics import YOLO

# ==========================================
# CONFIGURACIÓN PRINCIPAL
# ==========================================
# Ruta predeterminada desde donde se leerán los pesos (el modelo entrenado a usar)
MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt"

# Índice de la cámara que utilizará OpenCV (0 para la webcam principal nativa)
WEBCAM_INDEX = 0

# Tiempo (en segundos) que un trabajador debe infractar continuamente para considerarlo una falta
TIEMPO_CONFIRMACION = 2.0  

# Tiempo de espera (en segundos) tras haber registrado una falta antes de levantar otra (evita saturar los logs)
TIEMPO_ENFRIAMIENTO = 30.0 

# Nombre de la carpeta donde volcaremos todas las fotos de personas sin equipo
CARPETA_EVIDENCIAS = "evidencias"

# Intentamos crear la carpeta. exist_ok=True evita errores si la carpeta ya existe en el disco
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)

# NUEVO CSV FORMATO DATA SCIENCE (One-Hot Encoding)
# Construimos la ruta segura del archivo CSV usando el sistema de unión de rutas de 'os'
RUTA_CSV = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")

# Si el archivo aún no existe en el disco, significa que debemos crearlo y escribir sus cabeceras
if not os.path.exists(RUTA_CSV):
    # Abrimos/creamos el archivo en modo escritura ('w')
    with open(RUTA_CSV, mode='w', newline='') as archivo:
        # Iniciamos un escritor de CSV en ese archivo
        escritor = csv.writer(archivo)
        # Escribimos de inmediato la primera fila (los descriptores y variables formato One-Hot)
        escritor.writerow(["Fecha", "Hora", "ID_Persona", "Falta_Chaleco", "Falta_Casco", "Falta_Lentes", "Falta_Guantes", "Nombre_Foto"])

# Diccionario configurativo visual de etiquetas, agrupado por el string predictivo de YOLO
# Mapea (Nombre amigable, Color_BBR_para_OpenCV_en_Tupla)
CONFIGURACION_VISUAL = {
    'head_helmet': ("CON CASCO", (0, 255, 0)),        # Verde
    'head_nohelmet': ("SIN CASCO", (0, 0, 255)),      # Rojo
    'hand_glove': ("CON GUANTES", (0, 255, 0)),       # Verde
    'hand_noglove': ("SIN GUANTES", (0, 0, 255)),     # Rojo
    'face_mask': ("CON MASCARILLA", (0, 255, 0)),     # Verde
    'face_nomask': ("SIN MASCARILLA", (0, 0, 255)),   # Rojo
    'vest': ("CON CHALECO", (0, 255, 0)),             # Verde
    'glasses': ("CON LENTES", (0, 255, 0)),           # Verde
    'boots': ("BOTAS", (0, 255, 0)),                  # Verde
    'shoes': ("ZAPATOS", (0, 0, 255)),                # Rojo
    'person': ("PERSONA", (0, 255, 255))              # Amarillo
}

def get_optimal_device():
    """
    Evalúa la disponibilidad de chips dedicados (NVIDIA o chips de la familia M de Apple)
    y retorna el más potente. Si no hay ninguno, retorna procesamiento regular por CPU.
    """
    # Verificamos procesamiento de tensores en GPU Nvidia
    if torch.cuda.is_available(): return 'cuda'
    # Verificamos procesamiento en MPS (Metal Performance Shaders de Mac)
    if torch.backends.mps.is_available(): return 'mps'
    # Fallback seguro para cualquier computadora sin GPU
    return 'cpu'

def tiene_equipo(px1, py1, px2, py2, lista_coordenadas_equipo):
    """
    Función algorítmica espacial. Revisa si el punto central espacial de algún equipo de seguridad 
    (casco, chaleco, etc) cae DENTRO del cuadro delimitador de la persona.
    """
    # Iteramos sobre todos los equipos del tipo de interés detectados en la imagen
    for (cx1, cy1, cx2, cy2) in lista_coordenadas_equipo:
        # Obtenemos el punto central horizontal del equipo sumando x y dividiendo
        centro_x = (cx1 + cx2) // 2
        # Obtenemos el punto central vertical del equipo
        centro_y = (cy1 + cy2) // 2
        
        # Lógica espacial: ¿El punto (X, Y) del equipo está dentro de los límites de la Persona?
        if px1 < centro_x < px2 and py1 < centro_y < py2:
            # Si el equipo está físicamente sobre o dentro de la caja de esta persona, asumimos pertenencia
            return True
    
    # Si después de iterar todos los equipos de este tipo ninguno estaba en la persona, retorna False.
    return False

def main():
    """
    Función principal de integración y loop de procesamiento de video.
    """
    # Avisamos del boot de la UI del sistema
    print("Iniciando Sistema EPP (One-Hot Encoding CSV)...")
    
    # Definimos dónde va a correr la IA según la computadora (CPU, CUDA, MPS)
    device = get_optimal_device()
    
    # Carga de la red neuronal y pesos
    modelo = YOLO(MODEL_PATH)
    
    # Extraemos del modelo el diccionario de las clases para las cuales fue entrenado
    nombres_clases = modelo.names 

    # Invocamos la inicialización de captura de datos de video
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    # Si la webcam da error (bloqueo, falta de permisos), emitimos log y detenemos el código
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    # Creamos variables para medir timestamps y registrar las ventanas de tiempo entre reportes
    ultimo_reporte_tiempo = 0.0
    inicio_infraccion_tiempo = None

    # Avisamos al operador que ya estamos en ejecución
    print("\n¡Cámara activa! Presiona la tecla 'q' para salir.")

    # Loop Infinito del procesamiento del video 
    while True:
        # Extraemos 1 foto a la vez. Exito denota si falló el cable/conexión
        exito, frame = cap.read()
        
        # Rompemos limpiamente esta iteración si el frame llegó vacío
        if not exito: break
        
        # Duplicamos la foto antes de pintarle gráficos vectoriales, para tener evidencia "Limpia" que exportar si hay falta
        frame_limpio = frame.copy()
        
        # .track(persist=True) habilita en YOLO un algoritmo matemático para arrastrar temporalmente IDs consistentes
        # 'conf=0.6' es un umbral agresivo, descartamos falsos positivos menores de 60% de certeza.
        resultados = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
        
        # Recuperamos todas las predicciones de esta precisa imagen
        cajas = resultados[0].boxes

        # Listas de trabajo por frame. Las vaciaremos en cada vuelta.
        coordenadas_personas = []
        coordenadas_chalecos = []
        coordenadas_cascos = []
        coordenadas_lentes = []
        coordenadas_guantes = []

        # Diccionario One-Hot Default para iniciar la evaluación del frame asumiendo todo limpio. 
        # (Usaremos variables de tipo 1 / 0 para facilitar ciencia de datos)
        estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0, 'Guantes': 0}
        
        # Bandera de estado temporal que activaremos si alguien no tiene EPP
        hay_infraccion = False

        # Solo si procesamos una predicción con hallazgos (YOLO vio cosas):
        if cajas is not None and len(cajas) > 0:
            
            # Recorremos cada una de las delimitaciones halladas
            for caja in cajas:
                # Extraemos límites como enteros desde el tensor (necesario ya que cv2 odia float tensors al dibujar)
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                
                # Extraemos el identificador numérico de clase (ej. clase 0 puede ser 'persona')
                clase_id = int(caja.cls[0])
                
                # Traducimos de ID numérico a string legible usando los nombres del modelo (ej. "head_helmet")
                nombre_yolo = nombres_clases[clase_id]
                
                # Si .track le dio un ID a la persona, lo obtenemos. Si no, default a -1.
                track_id = int(caja.id[0]) if caja.id is not None else -1

                # Clasificamos la predicción y metemos sus dimensiones dentro de la lista correcta
                if nombre_yolo == 'person': coordenadas_personas.append((x1, y1, x2, y2, track_id))
                elif nombre_yolo == 'vest': coordenadas_chalecos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'head_helmet': coordenadas_cascos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'glasses': coordenadas_lentes.append((x1, y1, x2, y2))
                elif nombre_yolo == 'hand_glove': coordenadas_guantes.append((x1, y1, x2, y2))

                # Solicitamos el text label y color semántico de nuestro diccionario UI. En defecto usamos gris.
                texto_mostrar, color = CONFIGURACION_VISUAL.get(nombre_yolo, (nombre_yolo.upper(), (128, 128, 128)))
                
                # Particularización del UI si se detectó una persona: Modificamos el texto para incluir el Tracker ID
                if nombre_yolo == 'person' and track_id != -1:
                    texto_mostrar = f"PERSONA #{track_id}"

                # Pintamos el contorno cuadrado sobre el fotograma orginal con el color extraído
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Pintamos el texto etiquetando a qué pertenece la caja delimitadora por encima del borde superior de la caja
                cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Para cada persona detectada, analizaremos qué nivel de equipo posee para reportar
            for (px1, py1, px2, py2, p_id) in coordenadas_personas:
                # Recopilador de fallos de EPP para mostrar por pantalla
                lista_faltas_texto = []
                
                # Computamos variables binarias usando la lógica de pertenencia espacial (1 si NO lo tiene, 0 si SÍ lo tiene)
                falta_chaleco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos) else 0
                falta_casco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos) else 0
                falta_lentes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes) else 0
                falta_guantes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_guantes) else 0

                # Formateo verbal UI: Añadimos a la lista textos descriptivos de las faltas computadas
                if falta_chaleco: lista_faltas_texto.append("Chaleco")
                if falta_casco: lista_faltas_texto.append("Casco")
                if falta_lentes: lista_faltas_texto.append("Lentes")
                if falta_guantes: lista_faltas_texto.append("Guantes")
                
                # Si una persona se quedó sin cualquiera de sus partes obligatorias de protección personal
                if len(lista_faltas_texto) > 0:
                    # Avisamos al ciclo general que el frame actual califica como 'con infracciones'
                    hay_infraccion = True
                    
                    # Guardamos el estado exacto (One-Hot) y el ID para mandarlo al CSV en caso de cruzar el tiempo de confirmación
                    estado_infraccion = {
                        'ID': p_id, 'Chaleco': falta_chaleco, 'Casco': falta_casco, 
                        'Lentes': falta_lentes, 'Guantes': falta_guantes
                    }
                    
                    # UI Flotante para el trabajador en cuestión: Juntamos su lista de ausencias para pintársela de alarma
                    texto_faltas = "SIN: " + ", ".join(lista_faltas_texto)
                    
                    # Lo pintamos por debajo del lado superior de la Persona usando fuente color rojo (0,0,255)
                    cv2.putText(frame, texto_faltas, (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ==========================================
        # LÓGICA DE TIEMPOS Y ALERTAS
        # ==========================================
        # Extraemos timestamp del punto actual del procesador
        tiempo_actual = time.time()
        
        # Verificamos cuánto tiempo asincrónico ha pasado desde la última vez que insertamos una evidencia al CSV
        tiempo_desde_ultimo_reporte = tiempo_actual - ultimo_reporte_tiempo

        # Validación 1: El tiempo en enfriamiento estricto no se ha logrado aún
        if tiempo_desde_ultimo_reporte < TIEMPO_ENFRIAMIENTO:
            # Forzamos una variable restadora y dibujamos el UI de "PAUSA" para no atosigar al operador con multas repetitivas
            tiempo_restante = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_ultimo_reporte)
            cv2.putText(frame, f"SISTEMA EN PAUSA: {tiempo_restante}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # A pesar de que la red neuronal detecte faltas, este Null reseta el flag de alerta para no acumular el castigo en backgroud.
            inicio_infraccion_tiempo = None 
        
        else: # Validación 2: Salimos del enfriamiento. Sistemas de alerta completamente restaurados.
            # Imprimir al operador UI verde de que el sistema ya está escuchando nuevas contravenciones
            cv2.putText(frame, "SISTEMA ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Si la IA escaneó y confirmó una falta del trabajador evaluando el presente frame (estado infractor):
            if hay_infraccion:
                
                # Si esta persona apenas acaba de infractar ahora, iniciamos el cronómetro de la alerta temporal
                if inicio_infraccion_tiempo is None:
                    inicio_infraccion_tiempo = tiempo_actual
                
                # Validamos cuántas fracciones de tiempo continuo lleva amonestado
                segundos_infractando = tiempo_actual - inicio_infraccion_tiempo
                
                # ¿Ya pasaron suficientes (2) segundos continuos violando la norma para dictar el castigo?
                # (Esto mitiga falsos positivos fortuitos generados por pequeños desenfoques en la cámara o vibraciones breves)
                if segundos_infractando >= TIEMPO_CONFIRMACION:
                    
                    # Extraemos las fechas legibles para el File System de disco y el archivo de análisis CSV
                    ahora = datetime.now()
                    fecha_str = ahora.strftime("%Y-%m-%d")
                    hora_str = ahora.strftime("%H:%M:%S")
                    
                    # Nombramos la foto usando estampa temporal codificada para no sobreescribir evidencias
                    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}.jpg"
                    
                    # Unimos la ruta y el nombre (por ejemplo: "evidencias/falta_123000.jpg") validamente para Windows/Linux/Mac
                    ruta_foto = os.path.join(CARPETA_EVIDENCIAS, nombre_foto)
                    
                    # Volcamos `frame_limpio` que no traía recuadros en nuestro disco a modo de prueba pericial
                    cv2.imwrite(ruta_foto, frame_limpio)
                    
                    # ESCRITURA ONE-HOT EN CSV - Abrimos el archivo maestro en modo "append ('a')" para no borrar evidencias pasadas
                    with open(RUTA_CSV, mode='a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        
                        # Anexamos la fila con datos en tupla estructurada lista para su evaluación en Pandas/Modelamiento Data Science
                        escritor.writerow([
                            fecha_str, hora_str, 
                            estado_infraccion['ID'],
                            estado_infraccion['Chaleco'],
                            estado_infraccion['Casco'],
                            estado_infraccion['Lentes'],
                            estado_infraccion['Guantes'],
                            nombre_foto
                        ])
                    
                    # Informamos por stdout que el guardado transaccional se ha disparado correctamente
                    print(f"[{hora_str}] ¡INFRACCION! ID #{estado_infraccion['ID']} | Foto: {nombre_foto}")
                    
                    # Re-iniciamos el reloj de enfriamiento para mutear el sistema por la siguiente tanda de X segundos
                    ultimo_reporte_tiempo = tiempo_actual
                    
                    # Apagamos el seguimiento infractor de este evento para resetear el reloj cronometrado
                    inicio_infraccion_tiempo = None
                
                else:
                    # El trabajador no trae su EPP, pero aún le queda gracia antes del castigo (Temporizador de 2.0s a ~0.0s).
                    cv2.putText(frame, f"CONFIRMANDO FALTA... {int(TIEMPO_CONFIRMACION - segundos_infractando)}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # Si el frame evaluado pasó pruebas de EPP limpias, apagamos por completo el reloj que cronómetra posibles faltas 
                inicio_infraccion_tiempo = None

        # Desplegamos la versión intervenida gráficamente mediante el gestor visual de ventana
        cv2.imshow("Sistema EPP - Fase 3", frame)

        # Mecanismo de gracia interrumptiva. Capturamos el keycode del buffer. Si se presiona 'q', abandonamos limpieza.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Se instruye al CPU y la memoria soltar con delicadeza la instancia consumida de capturador WebCam
    cap.release()
    
    # Destrucción GUI explícita de todo el display de alto nivel
    cv2.destroyAllWindows()

# Entrypoint primario tipo script. Si fue llamado directo y no módulo secundario, ejecutamos la función núcleo.
if __name__ == "__main__":
    main()