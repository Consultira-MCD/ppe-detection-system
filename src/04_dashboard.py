import streamlit as st
import pandas as pd
import os
import cv2
import torch
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y VARIABLES
# ==========================================
st.set_page_config(page_title="Dashboard EPP", page_icon="🚧", layout="wide")

MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt"
WEBCAM_INDEX = 0

TIEMPO_CONFIRMACION = 2.0  
TIEMPO_ENFRIAMIENTO = 30.0 

CARPETA_EVIDENCIAS = "evidencias"
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)
RUTA_CSV = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")

CONFIGURACION_VISUAL = {
    'head_helmet': ("CON CASCO", (0, 255, 0)),
    'head_nohelmet': ("SIN CASCO", (0, 0, 255)),
    'hand_glove': ("CON GUANTES", (0, 255, 0)),
    'hand_noglove': ("SIN GUANTES", (0, 0, 255)),
    'face_mask': ("CON MASCARILLA", (0, 255, 0)),
    'face_nomask': ("SIN MASCARILLA", (0, 0, 255)),
    'vest': ("CON CHALECO", (0, 255, 0)),
    'glasses': ("CON LENTES", (0, 255, 0)),
    'boots': ("BOTAS", (0, 255, 0)),
    'shoes': ("ZAPATOS", (0, 0, 255)),
    'person': ("PERSONA", (0, 255, 255))
}

# ==========================================
# 2. FUNCIONES DEL SISTEMA
# ==========================================
@st.cache_resource
def cargar_modelo():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return YOLO(MODEL_PATH), device

def tiene_equipo(px1, py1, px2, py2, lista_coordenadas_equipo):
    for (cx1, cy1, cx2, cy2) in lista_coordenadas_equipo:
        centro_x = (cx1 + cx2) // 2
        centro_y = (cy1 + cy2) // 2
        if px1 < centro_x < px2 and py1 < centro_y < py2:
            return True
    return False

modelo, device = cargar_modelo()
nombres_clases = modelo.names

# ==========================================
# 3. INTERFAZ WEB (FRONTEND)
# ==========================================
st.title("🚧 Panel de Monitoreo EPP en Tiempo Real")
st.markdown("Sistema automatizado de detección de Equipo de Protección Personal.")

st.sidebar.header("⚙️ Configuración del Sistema")
fuente_video = st.sidebar.radio("Fuente de video:", ["Cámara Web en Vivo", "Video de Prueba (MP4)"])

# ¡NUEVO!: Botón interactivo para subir el video
archivo_video = None
if fuente_video == "Video de Prueba (MP4)":
    archivo_video = st.sidebar.file_uploader("Sube tu video corto aquí", type=['mp4', 'mov', 'avi'])

# --- SECCIÓN DE MÉTRICAS ---
st.subheader("📊 Historial de Infracciones")

contenedor_metricas = st.empty()

def actualizar_dashboard():
    with contenedor_metricas.container():
        if os.path.exists(RUTA_CSV):
            df = pd.read_csv(RUTA_CSV)
            if not df.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total de Infracciones", len(df))
                col2.metric("Última Infracción", str(df.iloc[-1]['Hora']))
                col3.metric("IDs Infractores", df['ID_Persona'].nunique())

                tabla_html = df.tail(10).to_html(index=False)
                st.markdown(tabla_html, unsafe_allow_html=True)
            else:
                st.warning("El archivo CSV existe, pero está vacío.")
        else:
            st.info("Aún no hay archivo de incidencias. Inicia el sistema para generar reportes.")

actualizar_dashboard()

# --- SECCIÓN DE VIDEO EN VIVO ---
st.markdown("---")
st.subheader("📹 Transmisión de Seguridad")

run_camera = st.checkbox("🔴 Activar Monitoreo")
marco_video = st.empty()

# ==========================================
# 4. BUCLE DE PROCESAMIENTO DE VIDEO
# ==========================================
if run_camera:
    # ¡NUEVA LÓGICA PARA LEER EL VIDEO SUBIDO!
    if fuente_video == "Cámara Web en Vivo":
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    elif fuente_video == "Video de Prueba (MP4)":
        if archivo_video is not None:
            # Guardamos el archivo temporalmente para que OpenCV lo pueda leer
            ruta_temp = os.path.join(CARPETA_EVIDENCIAS, "temp_video.mp4")
            with open(ruta_temp, "wb") as f:
                f.write(archivo_video.read())
            cap = cv2.VideoCapture(ruta_temp)
        else:
            st.warning("⚠️ Por favor sube un archivo de video en el menú lateral para continuar.")
            st.stop()

    ultimo_reporte_tiempo = 0.0
    inicio_infraccion_tiempo = None

    while run_camera:
        exito, frame = cap.read()
        if not exito:
            st.info("Fin de la transmisión del video.")
            break

        frame_limpio = frame.copy()
        resultados = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
        cajas = resultados[0].boxes

        coordenadas_personas = []
        coordenadas_chalecos, coordenadas_cascos = [], []
        coordenadas_lentes, coordenadas_guantes = [], []
        estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0, 'Guantes': 0}
        hay_infraccion = False

        if cajas is not None and len(cajas) > 0:
            for caja in cajas:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                clase_id = int(caja.cls[0])
                nombre_yolo = nombres_clases[clase_id]
                track_id = int(caja.id[0]) if caja.id is not None else -1

                if nombre_yolo == 'person': coordenadas_personas.append((x1, y1, x2, y2, track_id))
                elif nombre_yolo == 'vest': coordenadas_chalecos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'head_helmet': coordenadas_cascos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'glasses': coordenadas_lentes.append((x1, y1, x2, y2))
                elif nombre_yolo == 'hand_glove': coordenadas_guantes.append((x1, y1, x2, y2))

                texto_mostrar, color = CONFIGURACION_VISUAL.get(nombre_yolo, (nombre_yolo.upper(), (128, 128, 128)))
                if nombre_yolo == 'person' and track_id != -1:
                    texto_mostrar = f"PERSONA #{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for (px1, py1, px2, py2, p_id) in coordenadas_personas:
                lista_faltas_texto = []
                falta_chaleco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos) else 0
                falta_casco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos) else 0
                falta_lentes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes) else 0
                falta_guantes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_guantes) else 0

                if falta_chaleco: lista_faltas_texto.append("Chaleco")
                if falta_casco: lista_faltas_texto.append("Casco")
                if falta_lentes: lista_faltas_texto.append("Lentes")
                if falta_guantes: lista_faltas_texto.append("Guantes")
                
                if len(lista_faltas_texto) > 0:
                    hay_infraccion = True
                    estado_infraccion = {
                        'ID': p_id, 'Chaleco': falta_chaleco, 'Casco': falta_casco, 
                        'Lentes': falta_lentes, 'Guantes': falta_guantes
                    }
                    texto_faltas = "SIN: " + ", ".join(lista_faltas_texto)
                    cv2.putText(frame, texto_faltas, (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        tiempo_actual = time.time()
        tiempo_desde_ultimo_reporte = tiempo_actual - ultimo_reporte_tiempo

        if tiempo_desde_ultimo_reporte < TIEMPO_ENFRIAMIENTO:
            tiempo_restante = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_ultimo_reporte)
            cv2.putText(frame, f"SISTEMA EN PAUSA: {tiempo_restante}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            inicio_infraccion_tiempo = None 
        else:
            cv2.putText(frame, "SISTEMA ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if hay_infraccion:
                if inicio_infraccion_tiempo is None:
                    inicio_infraccion_tiempo = tiempo_actual
                segundos_infractando = tiempo_actual - inicio_infraccion_tiempo
                
                if segundos_infractando >= TIEMPO_CONFIRMACION:
                    ahora = datetime.now()
                    fecha_str = ahora.strftime("%Y-%m-%d")
                    hora_str = ahora.strftime("%H:%M:%S")
                    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}.jpg"
                    ruta_foto = os.path.join(CARPETA_EVIDENCIAS, nombre_foto)
                    
                    cv2.imwrite(ruta_foto, frame_limpio)
                    with open(RUTA_CSV, mode='a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        escritor.writerow([
                            fecha_str, hora_str, estado_infraccion['ID'],
                            estado_infraccion['Chaleco'], estado_infraccion['Casco'],
                            estado_infraccion['Lentes'], estado_infraccion['Guantes'], nombre_foto
                        ])
                    
                    actualizar_dashboard()
                    
                    ultimo_reporte_tiempo = tiempo_actual
                    inicio_infraccion_tiempo = None
                else:
                    cv2.putText(frame, f"CONFIRMANDO FALTA... {int(TIEMPO_CONFIRMACION - segundos_infractando)}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                inicio_infraccion_tiempo = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marco_video.image(frame_rgb, channels="RGB", use_column_width=True)

        # Pequeña pausa para que el MP4 se reproduzca a velocidad normal
        if fuente_video == "Video de Prueba (MP4)":
            time.sleep(0.03)

    cap.release()
else:
    st.info("Haz clic en la casilla de arriba para encender la cámara y procesar el video.")