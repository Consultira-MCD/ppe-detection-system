"""
Script del Panel de Monitoreo Web (Dashboard) interactivo.
Utiliza la biblioteca Streamlit para construir una interfaz gráfica en el navegador.
Permite seleccionar la fuente de video (Webcam o carga de MP4 local), carga el modelo
YOLOv8 entrenado para detectar EPP, y consolida visualmente los análisis de las cajas 
delimitadoras en tiempo real. También lee y muestra los KPIs desde el archivo CSV 
(formateado como One-Hot Encoding).
"""

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
# Configura el título y layout de la página de Streamlit
st.set_page_config(page_title="Dashboard EPP", page_icon="🚧", layout="wide")

# Rutas y variables de configuración de la arquitectura
MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt"
WEBCAM_INDEX = 0
TIEMPO_CONFIRMACION = 2.0  
TIEMPO_ENFRIAMIENTO = 30.0 

# Creación de directorio para almacenar evidencias
CARPETA_EVIDENCIAS = "evidencias"
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)
RUTA_CSV = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")

# Diccionario de traducción y asignación de colores (BGR) para detecciones
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
    """
    Carga el modelo YOLO en memoria. Usa @st.cache_resource para evitar 
    recargas constantes en Streamlit.
    """
    # Preferencia por procesador MPS (Mac) si está disponible, caso contrario CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return YOLO(MODEL_PATH), device

def tiene_equipo(px1, py1, px2, py2, lista_coordenadas_equipo):
    """
    Verifica geométricamente si un integrante del equipo (ej. casco) 
    se encuentra físicamente dentro del recuadro del sujeto detectado.
    """
    for (cx1, cy1, cx2, cy2) in lista_coordenadas_equipo:
        # Cálculo del centroide
        centro_x = (cx1 + cx2) // 2
        centro_y = (cy1 + cy2) // 2
        # Evaluación de los límites
        if px1 < centro_x < px2 and py1 < centro_y < py2:
            return True
    return False

# Inicialización dinámica en cache del sistema de IA
modelo, device = cargar_modelo()
nombres_clases = modelo.names

# ==========================================
# 3. INTERFAZ WEB (FRONTEND)
# ==========================================
# Encabezados de la aplicación Dashboard
st.title("Panel de Monitoreo EPP en Tiempo Real")
st.markdown("Sistema automatizado de detección de Equipo de Protección Personal.")

# Barra lateral de configuración y operativas
st.sidebar.header("⚙️ Configuración del Sistema")
fuente_video = st.sidebar.radio("Fuente de video:", ["Cámara Web en Vivo", "Video de Prueba (MP4)"])

archivo_video = None
# Opción de carga de archivo si es seleccionado por el operador
if fuente_video == "Video de Prueba (MP4)":
    archivo_video = st.sidebar.file_uploader("Sube tu video corto aquí", type=['mp4', 'mov', 'avi'])

# --- SECCIÓN 1: VIDEO EN VIVO ---
st.subheader("Transmisión de Seguridad")
run_camera = st.checkbox(" Activar Monitoreo en Vivo")

# Contenedor dinámico de Streamlit para el Canvas
marco_video = st.empty()

# --- SECCIÓN 2: MÉTRICAS Y GRÁFICAS ---
st.markdown("---")
st.subheader("Análisis e Historial de Infracciones")

# Contenedores dinámicos para KPIs top-level
contenedor_tarjetas = st.empty()

# Columnas flexibles
col_izq, col_der = st.columns(2)
with col_izq:
    contenedor_grafica = st.empty()
with col_der:
    contenedor_tabla = st.empty()

def actualizar_dashboard():
    """
    Lee el archivo local CSV y actualiza las KPIs mostradas.
    """
    if os.path.exists(RUTA_CSV):
        # Leemos el CSV asignando explícitamente las cabeceras
        df = pd.read_csv(RUTA_CSV, names=['Fecha', 'Hora', 'ID_Persona', 'Chaleco', 'Casco', 'Lentes', 'Guantes', 'Nombre_Foto'], header=0)
        
        # Conversión a números validada
        df['Chaleco'] = pd.to_numeric(df['Chaleco'], errors='coerce').fillna(0)
        df['Casco'] = pd.to_numeric(df['Casco'], errors='coerce').fillna(0)
        df['Lentes'] = pd.to_numeric(df['Lentes'], errors='coerce').fillna(0)
        df['Guantes'] = pd.to_numeric(df['Guantes'], errors='coerce').fillna(0)

        # Si el DataFrame histórico posee filas:
        if not df.empty:
            # Renderizado de Tarjetas Informativas
            with contenedor_tarjetas.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Total de Infracciones", len(df))
                c2.metric("Última Infracción", str(df.iloc[-1]['Hora']))
                c3.metric("IDs Diferentes", df['ID_Persona'].nunique())

            # Renderizado de Gráficas por código HTML/CSS inyectado
            with contenedor_grafica.container():
                st.markdown("**Infracciones más comunes**")
                totales = {
                    'Sin Chaleco': int(df['Chaleco'].sum()),
                    'Sin Casco': int(df['Casco'].sum()),
                    'Sin Lentes': int(df['Lentes'].sum()),
                    'Sin Guantes': int(df['Guantes'].sum())
                }
                
                # Obtención de valor máximo paramétrico
                max_val = max(totales.values()) if max(totales.values()) > 0 else 1
                
                # Construcción e iteración del Bloque en Líneas de UI
                html_barras = ""
                for equipo, cantidad in totales.items():
                    porcentaje = int((cantidad / max_val) * 100)
                    html_barras += f"""
                    <div style="margin-bottom: 12px; font-family: sans-serif;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span>{equipo}</span>
                            <strong>{cantidad}</strong>
                        </div>
                        <div style="background-color: #333; border-radius: 4px; width: 100%; height: 20px;">
                            <div style="background-color: #FF4B4B; width: {porcentaje}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """
                st.markdown(html_barras, unsafe_allow_html=True)

            # Renderizado de la respectiva Tabla Estilizada
            with contenedor_tabla.container():
                st.markdown("**Base de Datos Completa (Desliza para ver más)**")
                # Invierte el DataFrame para histórico descendente
                df_invertido = df.iloc[::-1] 
                
                # Descompone el DF puro en tablas web (TR/TDs)
                tabla_cruda = df_invertido.to_html(index=False, border=0).replace('\n', '')
                
                # Estilos purificados
                estilo_tabla = """
                <style>
                    .tabla-scroll table { width: 100%; text-align: left; color: white; border-collapse: collapse; font-size: 14px;}
                    .tabla-scroll th { background-color: #262730; padding: 10px; border-bottom: 2px solid #FF4B4B;}
                    .tabla-scroll td { padding: 8px; border-bottom: 1px solid #333; }
                </style>
                """.replace('\n', '')
                
                # Impresión responsiva con SCROLL
                html_final = f'{estilo_tabla}<div class="tabla-scroll" style="max-height: 250px; overflow-y: auto;">{tabla_cruda}</div>'
                st.markdown(html_final, unsafe_allow_html=True)

# Actualización inicial forzada pre-carga
actualizar_dashboard()

# ==========================================
# 4. BUCLE DE PROCESAMIENTO DE VIDEO
# ==========================================
if run_camera:
    # Captura periférica
    if fuente_video == "Cámara Web en Vivo":
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    # Captura pre-grabada
    elif fuente_video == "Video de Prueba (MP4)":
        if archivo_video is not None:
            # Requisito binario para OpenCV: Grabado local para parsear video
            ruta_temp = os.path.join(CARPETA_EVIDENCIAS, "temp_video.mp4")
            with open(ruta_temp, "wb") as f:
                f.write(archivo_video.read())
            cap = cv2.VideoCapture(ruta_temp)
        else:
            st.warning("Por favor sube un archivo de video en el menú lateral para continuar.")
            st.stop()

    # Inicialización de cronómetros
    ultimo_reporte_tiempo = 0.0
    inicio_infraccion_tiempo = None

    # Thread activo
    while run_camera:
        exito, frame = cap.read()
        if not exito:
            st.info("Fin de la transmisión del video.")
            break

        # Aislamiento original para reporte sin recuadros superpuestos
        frame_limpio = frame.copy()
        
        # Red de tracking y confidence con el dispositivo previamente listado
        resultados = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
        cajas = resultados[0].boxes

        # Reseteo relacional para procesarse el Frame Cero
        coordenadas_personas = []
        coordenadas_chalecos, coordenadas_cascos = [], []
        coordenadas_lentes, coordenadas_guantes = [], []
        
        # Valores lógicos nativos infractores con base nula 
        estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0, 'Guantes': 0}
        hay_infraccion = False

        if cajas is not None and len(cajas) > 0:
            for caja in cajas:
                # Decodificamos cartesianamente los extremos
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                clase_id = int(caja.cls[0])
                nombre_yolo = nombres_clases[clase_id]
                
                # Tracking persistente (ByteTrack internal algos)
                track_id = int(caja.id[0]) if caja.id is not None else -1

                # Discriminamos clase por variables
                if nombre_yolo == 'person': coordenadas_personas.append((x1, y1, x2, y2, track_id))
                elif nombre_yolo == 'vest': coordenadas_chalecos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'head_helmet': coordenadas_cascos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'glasses': coordenadas_lentes.append((x1, y1, x2, y2))
                elif nombre_yolo == 'hand_glove': coordenadas_guantes.append((x1, y1, x2, y2))

                # Resolución visual desde Settings Diccionario (Tupla color)
                texto_mostrar, color = CONFIGURACION_VISUAL.get(nombre_yolo, (nombre_yolo.upper(), (128, 128, 128)))
                
                # Identificativo exclusivo para Persona ID
                if nombre_yolo == 'person' and track_id != -1:
                    texto_mostrar = f"PERSONA #{track_id}"

                # Sobre dibujo al lienzo del Frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Validación de equipo para infractores Human-Clustered
            for (px1, py1, px2, py2, p_id) in coordenadas_personas:
                lista_faltas_texto = []
                
                # Traduce Mapeo Espacial de EPP vs Persona = One-Hot Encoding Score Predictivo
                falta_chaleco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos) else 0
                falta_casco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos) else 0
                falta_lentes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes) else 0
                falta_guantes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_guantes) else 0

                # Etiquetas literales listadas
                if falta_chaleco: lista_faltas_texto.append("Chaleco")
                if falta_casco: lista_faltas_texto.append("Casco")
                if falta_lentes: lista_faltas_texto.append("Lentes")
                if falta_guantes: lista_faltas_texto.append("Guantes")
                
                # Penalización afirmativa
                if len(lista_faltas_texto) > 0:
                    hay_infraccion = True
                    estado_infraccion = {
                        'ID': p_id, 'Chaleco': falta_chaleco, 'Casco': falta_casco, 
                        'Lentes': falta_lentes, 'Guantes': falta_guantes
                    }
                    texto_faltas = "SIN: " + ", ".join(lista_faltas_texto)
                    # Advertencia Rojo Flotante
                    cv2.putText(frame, texto_faltas, (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Seguimiento del Cronómetro
        tiempo_actual = time.time()
        tiempo_desde_ultimo_reporte = tiempo_actual - ultimo_reporte_tiempo

        # Validación Anti-Spam de Multas Excesivas
        if tiempo_desde_ultimo_reporte < TIEMPO_ENFRIAMIENTO:
            tiempo_restante = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_ultimo_reporte)
            cv2.putText(frame, f"SISTEMA EN PAUSA: {tiempo_restante}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            inicio_infraccion_tiempo = None 
        else:
            cv2.putText(frame, "SISTEMA ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Falla en progreso activo
            if hay_infraccion:
                if inicio_infraccion_tiempo is None:
                    inicio_infraccion_tiempo = tiempo_actual
                
                # Validador continuo de persistencia
                segundos_infractando = tiempo_actual - inicio_infraccion_tiempo
                
                if segundos_infractando >= TIEMPO_CONFIRMACION:
                    ahora = datetime.now()
                    fecha_str = ahora.strftime("%Y-%m-%d")
                    hora_str = ahora.strftime("%H:%M:%S")
                    
                    # Fotografía JPG Pericial
                    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}.jpg"
                    ruta_foto = os.path.join(CARPETA_EVIDENCIAS, nombre_foto)
                    
                    cv2.imwrite(ruta_foto, frame_limpio)
                    
                    # Conector FileSystem (Append a=CSV)
                    with open(RUTA_CSV, mode='a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        escritor.writerow([
                            fecha_str, hora_str, estado_infraccion['ID'],
                            estado_infraccion['Chaleco'], estado_infraccion['Casco'],
                            estado_infraccion['Lentes'], estado_infraccion['Guantes'], nombre_foto
                        ])
                    
                    # Invocación Backend Dashboard Refresher
                    actualizar_dashboard()
                    ultimo_reporte_tiempo = tiempo_actual
                    inicio_infraccion_tiempo = None
                else:
                    cv2.putText(frame, f"CONFIRMANDO FALTA... {int(TIEMPO_CONFIRMACION - segundos_infractando)}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                inicio_infraccion_tiempo = None

        # Convertr BGR (OpenCV Defaults) a RGB (Streamlit Web Defaults)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Superposición de la Matriz Fotograma para Streamlit
        marco_video.image(frame_rgb, channels="RGB", use_column_width=True)

        # Retención Natural (Evita que el stream MP4 corra muy de prisa en Macs nuevas)
        if fuente_video == "Video de Prueba (MP4)":
            time.sleep(0.03)

    # Eliminación de Puntero IO
    cap.release()
else:
    st.info("Haz clic en la casilla de arriba para encender la cámara y procesar el video.")