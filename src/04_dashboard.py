"""
Fase 1 del proyecto de la consultoría.

Script del Panel de Monitoreo Web (Dashboard) interactivo.
Utiliza la biblioteca Streamlit para construir una interfaz gráfica en el navegador.
Permite seleccionar la fuente de video (Webcam o carga de MP4 local), carga el modelo
YOLOv8 entrenado para detectar EPP, y consolida visualmente los análisis de las cajas 
delimitadoras en tiempo real. También lee y muestra los KPIs desde el archivo CSV 
(formateado como One-Hot Encoding) conforme se van registrando infracciones, 
apoyándose en componentes HTML/CSS responsivos integrados de forma nativa.
"""

# Importamos Streamlit para crear y gestionar la interfaz web del dashboard analítico
import streamlit as st

# Importamos Pandas para estructurar los datos del CSV en DataFrames y manipular métricas tabulares
import pandas as pd

# Importamos os para crear directorios y componer rutas de archivos independientemente 
# del sistema operativo
import os

# Importamos OpenCV (cv2) para capturar el feed de video y dibujar recuadros superpuestos
import cv2

# Importamos PyTorch para manejar las operaciones matemáticas y la delegación del acelerador 
# de hardware (GPU/MPS/CPU)
import torch

# Importamos time para sincronizar el bucle de renderizado de video y los tiempos de 
# enfriamiento del modelo
import time

# Importamos csv para almacenar de manera estructurada y persistente las incidencias 
# detectadas en disco
import csv

# Importamos datetime para etiquetar con precisión temporal las infracciones detectadas 
# y los nombres de las capturas
from datetime import datetime

# Importamos la clase principal YOLO de ultralytics para orquestar la inferencia en tiempo real
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y VARIABLES
# ==========================================
# Configuración inicial del layout del dashboard para usar todo el ancho de la pantalla
st.set_page_config(page_title="Dashboard EPP", page_icon="🚧", layout="wide")

# Ruta predefinida hacia los pesos finales del algoritmo predictivo
MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt"

# Índice estándar para hardware de captura (Webcam principal por defecto)
WEBCAM_INDEX = 0

# Límite de persistencia (en segundos) requerido para dictaminar formalmente una infracción
TIEMPO_CONFIRMACION = 2.0  

# Límite de inactividad de reportes tras emitir una penalización a fin de evitar saturación de logs
TIEMPO_ENFRIAMIENTO = 30.0 

# Directorio que resguardará los recortes fotográficos y el archivo tabular de respaldos
CARPETA_EVIDENCIAS = "evidencias"

# Creación en sistema de archivos en caso de no existir previamente la carpeta de evidencias
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)

# Generación del string completo del archivo CSV para el almacenamiento en One-Hot Encoding
RUTA_CSV = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")

# Diccionario integral para traducir las clases predichas por YOLO en descripciones 
# y colores en formato BGR
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

# ==========================================
# 2. FUNCIONES DEL SISTEMA
# ==========================================

# Estabilizamos la carga del modelo para evitar recargas excesivas al re-ejecutar el 
# árbol del UI de Streamlit
@st.cache_resource
def cargar_modelo():
    """Inicializa estáticamente la IA en memoria e identifica el chip de aceleración 
    más óptimo a usar."""
    # Detectamos soporte de Metal Performance Shaders (Apple) en preferencia frente a CPU básico
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Retornamos la instanciación principal de la topología junto con su aceleradora 
    # computacional de hardware
    return YOLO(MODEL_PATH), device

def tiene_equipo(px1, py1, px2, py2, lista_coordenadas_equipo):
    """
    Función de asociación espacial simple: Evalúa matemáticamente si el centro geométrico del 
    EPP detectado se encuentra físicamente comprendido dentro de la caja limitadora 
    de una Persona en pantalla.
    """
    # Recorremos cada tupla en el arreglo del Equipo detectado
    for (cx1, cy1, cx2, cy2) in lista_coordenadas_equipo:
        
        # Computamos el centro geométrico espacial dividiendo las proyecciones cartesianas
        centro_x = (cx1 + cx2) // 2
        centro_y = (cy1 + cy2) // 2
        
        # Validamos si el punto EPP X,Y está encerrado entre los extremos opuestos 
        # espaciales de la silueta de la persona
        if px1 < centro_x < px2 and py1 < centro_y < py2:
            return True
            
    # Retorna falso de manera unívoca si la silueta particular de esa persona no 
    # englobó ningún punto del equipo
    return False

# Disparamos la carga temprana de los Pesos Neuronales aprovechando el decorador @st.cache_resource
modelo, device = cargar_modelo()

# Mantenemos las jerarquías nominales originales detectadas
nombres_clases = modelo.names

# ==========================================
# 3. INTERFAZ WEB (FRONTEND)
# ==========================================
# Declaramos la cabecera prominente y profesional dictaminando el propósito del software
st.title("Panel de Monitoreo EPP en Tiempo Real")
st.markdown("Sistema automatizado de detección de Equipo de Protección Personal.")

st.sidebar.header("⚙️ Configuración del Sistema")
fuente_video = st.sidebar.radio("Fuente de video:", ["Cámara Web en Vivo", "Video de Prueba (MP4)"])

# Condicionalmente mostramos u ocultamos el componente drag'n drop de Streamlit 
# acorde con el tipo de fuente solicitada
archivo_video = None
if fuente_video == "Video de Prueba (MP4)":
    archivo_video = st.sidebar.file_uploader("Sube tu video corto aquí", type=['mp4', 'mov', 'avi'])

# --- SECCIÓN 1: VIDEO EN VIVO ---
# Componentes top-level relacionados directamente a la visualización ininterrumpida 
# de video trackeable
st.subheader("Transmisión de Seguridad")

# Interruptor de arranque booleano de la maquinaria para que Streamlit sepa en qué 
# momento delegar ciclos CPU a YOLO
run_camera = st.checkbox("Activar Monitoreo en Vivo")

# Se inicializa un puerto vacío que Streamlit iterará posteriormente en su flujo 
# renderizado con las fotos convertidas de YOLO
marco_video = st.empty()


# --- SECCIÓN 2: MÉTRICAS Y GRÁFICAS ---
# Separador cosmético para fraccionar elegantemente la UI 
st.markdown("---")
st.subheader("Análisis e Historial de Infracciones")

# Declaración del contenedor de mayor jerarquía para el KPI resumido
contenedor_tarjetas = st.empty()

# Layout del grid en columnas responsivas: Columna izq para barras y Columna derecha para Tabla cruda
col_izq, col_der = st.columns(2)
with col_izq:
    contenedor_grafica = st.empty()
with col_der:
    contenedor_tabla = st.empty()

def actualizar_dashboard():
    """ 
    Proceso iterativo backend invocado modularmente tras cada infracción comprobada, 
    el cual lee el volcado CSV presente a fin de re-renderizar componentes de inteligencia 
    de negocio como gráficas y acumulables.
    """
    if os.path.exists(RUTA_CSV):
        # Lectura robusta mediante Pandas forzando intencionalmente cabeceras hardcodeadas 
        # acorde al formato CSV esperado
        df = pd.read_csv(RUTA_CSV, names=['Fecha', 'Hora', 'ID_Persona', 'Chaleco', 'Casco', 'Lentes', 'Guantes', 'Nombre_Foto'], header=0)
        
        # Normalizador de limpieza para garantizar integridad computacional frente a datos nulos: 
        # forzar parseo estadístico puro.
        df['Chaleco'] = pd.to_numeric(df['Chaleco'], errors='coerce').fillna(0)
        df['Casco'] = pd.to_numeric(df['Casco'], errors='coerce').fillna(0)
        df['Lentes'] = pd.to_numeric(df['Lentes'], errors='coerce').fillna(0)
        df['Guantes'] = pd.to_numeric(df['Guantes'], errors='coerce').fillna(0)

        # Confirmación anti Null-Reference si el Excel poseyera esquemas y ceros registros
        if not df.empty:
            
            # Bloque 1: Consolidado Superior de Tres Cartillas tipo Tablero de Control Ejecutivo
            with contenedor_tarjetas.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Total de Infracciones", len(df))
                c2.metric("Última Infracción", str(df.iloc[-1]['Hora']))
                c3.metric("IDs Diferentes", df['ID_Persona'].nunique())

            # Bloque 2: Gráfica de Barras Dinámica basada en componentes Flex de CSS 
            # (Renderizado Liviano y customizado)
            with contenedor_grafica.container():
                st.markdown("**Infracciones más comunes**")
                
                # Consolidación Agregada vía SUM pandas en diccionarios para reuso html
                totales = {
                    'Sin Chaleco': int(df['Chaleco'].sum()),
                    'Sin Casco': int(df['Casco'].sum()),
                    'Sin Lentes': int(df['Lentes'].sum()),
                    'Sin Guantes': int(df['Guantes'].sum())
                }
                
                # Escalado porcentual normalizado determinando el valor pico de la gráfica
                max_val = max(totales.values()) if max(totales.values()) > 0 else 1
                
                # Integrador iterativo de String CSS/HTML. Por norma de seguridad HTML, 
                # cada barra crece y se dibuja on-the-fly
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
                
                # Renderiza en cascada bajo etiqueta 'Segura' (unsafe_allow_html) de forma 
                # controlada y sin dependencias JS
                st.markdown(html_barras, unsafe_allow_html=True)
                 
            # Bloque 3: Tabla cruda con diseño CSS mejorado y comportamiento reactivo 
            # 'Overflow / Scroll'
            with contenedor_tabla.container():
                st.markdown("**Base de Datos Completa (Desliza para ver más)**")
                
                # Transformación de Indexado para inyección FILO (First-In-Last-Out style view) 
                # para la interfaz web.
                df_invertido = df.iloc[::-1]
                
                # Generador de marcado HTML crudo nativo de Pandas con optimizaciones de espacio 
                # visual
                tabla_cruda = df_invertido.to_html(index=False, border=0).replace('\n', '')
                
                # Style blocks parametrizados de CSS que otorgan apariencia Dark Theme en las
                # tabulaciones base
                estilo_tabla = """
                <style>
                    .tabla-scroll table { width: 100%; text-align: left; color: white; border-collapse: collapse; font-size: 14px;}
                    .tabla-scroll th { background-color: #262730; padding: 10px; border-bottom: 2px solid #FF4B4B;}
                    .tabla-scroll td { padding: 8px; border-bottom: 1px solid #333; }
                </style>
                """.replace('\n', '')
                
                # Envoltura final permitiendo 250 pixeles de contenedor scrollable en Y a nivel 
                # de documento sin afectar UX primaria 
                html_final = f'{estilo_tabla}<div class="tabla-scroll" style="max-height: 250px; overflow-y: auto;">{tabla_cruda}</div>'
                st.markdown(html_final, unsafe_allow_html=True)

# Evaluación inicial estática obligatoria del Dashboard pre-computación de Video
actualizar_dashboard()

# ==========================================
# 4. BUCLE DE PROCESAMIENTO DE VIDEO
# ==========================================
# Disparador condicionado exclusivo de interacción directa de operador Streamlit checkbox 
# 'run_camera'
if run_camera:
    
    # Derivación a subsistema en vivo V4L2 o CoreMedia
    if fuente_video == "Cámara Web en Vivo":
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        
    # Derivación para el parseo y descompresión de clip multimedia pre-guardado
    elif fuente_video == "Video de Prueba (MP4)":
        
        # Blindaje Streamlit: Garantizamos integridad temporal del Binario Upload antes de 
        # abrirlo vía C++ bindings (OpenCV)
        if archivo_video is not None:
            # Creación del artefacto volátil con el que consumiremos el encoder local 
            # ffmpeg/gstreamer
            ruta_temp = os.path.join(CARPETA_EVIDENCIAS, "temp_video.mp4")
            with open(ruta_temp, "wb") as f:
                f.write(archivo_video.read())
            cap = cv2.VideoCapture(ruta_temp)
        else:
            # Manejo preventivo para interrumpir la cascada gráfica si el usuario 
            # activó procesado antes de cargar el insumo
            st.warning("Por favor sube un archivo de video en el menú lateral para continuar.")
            st.stop()

    # Reajuste local de variables de seguimiento de infractores 
    # (Cronómetros nulos iniciales de la sesión)
    ultimo_reporte_tiempo = 0.0
    inicio_infraccion_tiempo = None

    # Iterador secuencial perpetuo asociado a la vida ininterrumpida de nuestro toggle bool 
    # Streamlit
    while run_camera:
        # Petición asertiva para extracción continua del bitstream fotogramático
        exito, frame = cap.read()
        
        # Liberación suave y controlada (Break) ante un Fin de archivo en MP4 o rotura 
        # de conexión USB
        if not exito:
            st.info("Fin de la transmisión del video.")
            break

        # Aislamiento puro de los píxeles antes de cualquier dibujado sobre ellos para 
        # preservar validez pericial
        frame_limpio = frame.copy()
        
        # Orquestación Neural en cascada. Track persite memoria algorítmica y Confidence 
        # garantiza severidad predictiva
        resultados = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
        cajas = resultados[0].boxes

        # Matrices limpiadas por clícker temporal para no fusionar datos vectoriales 
        # inter-fotograma
        coordenadas_personas = []
        coordenadas_chalecos, coordenadas_cascos = [], []
        coordenadas_lentes, coordenadas_guantes = [], []
        
        # Esquematización Base Line Cero Default del infractor. (Usando la lógica One-Hot 
        # Science donde 0=Cumple y 1=Falla)
        estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0, 'Guantes': 0}
        
        # Variable centinela del flujo Frame a procesar
        hay_infraccion = False

        # Solo si se han procesado vectores detectados en esta pasada:
        if cajas is not None and len(cajas) > 0:
            
            # Sub-iteramos la matrix devuelta por el Tensor y mapeamos a Int nativo 
            # listados BBR cartesianos
            for caja in cajas:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                
                # Interceptamos la clase de catálogo para referenciar en cadena 
                # "Nombres" vs id número predictivo puro
                clase_id = int(caja.cls[0])
                nombre_yolo = nombres_clases[clase_id]
                
                # Explotación de Persistencia DeepSort/ByteTrack provisto nativo en la línea 
                # .track de ultralytics
                track_id = int(caja.id[0]) if caja.id is not None else -1

                # Discriminador lógico de volcado a memoria relacional del array 
                # respectivo del sujeto / equipo
                if nombre_yolo == 'person': coordenadas_personas.append((x1, y1, x2, y2, track_id))
                elif nombre_yolo == 'vest': coordenadas_chalecos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'head_helmet': coordenadas_cascos.append((x1, y1, x2, y2))
                elif nombre_yolo == 'glasses': coordenadas_lentes.append((x1, y1, x2, y2))
                elif nombre_yolo == 'hand_glove': coordenadas_guantes.append((x1, y1, x2, y2))

                # Resolución visual consultando el Diccionario Color-Palette Constante
                texto_mostrar, color = CONFIGURACION_VISUAL.get(nombre_yolo, (nombre_yolo.upper(), (128, 128, 128)))
                
                # Extrapolamos nombre y texto especial si hablamos de siluetas 
                # Humanas con Tracking ID exitoso
                if nombre_yolo == 'person' and track_id != -1:
                    texto_mostrar = f"PERSONA #{track_id}"

                # Sobre-escrituras directas al lienzo matriz del Fotograma (Frame). 
                # Rectángulo envolvente y Header Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Para cada ser humano válido hallado en los límites de este Frame, 
            # calculamos faltantes de salud ocupacional HSE
            for (px1, py1, px2, py2, p_id) in coordenadas_personas:
                
                # Receptor string acumulativo utilizado estrictamente para pintar UI en 
                # rojo a la persona respectiva
                lista_faltas_texto = []
                
                # Inyección del Boolean Inverso en el OneHotEncoding 
                # (Si _No_ la función "tiene_equipo" retorna True -> entonces vale 1 "Fallo")
                falta_chaleco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos) else 0
                falta_casco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos) else 0
                falta_lentes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes) else 0
                falta_guantes = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_guantes) else 0

                # Append manual iterativo para construir enunciado semántico en español 
                # de la alarma en la capa del usuario
                if falta_chaleco: lista_faltas_texto.append("Chaleco")
                if falta_casco: lista_faltas_texto.append("Casco")
                if falta_lentes: lista_faltas_texto.append("Lentes")
                if falta_guantes: lista_faltas_texto.append("Guantes")
                
                # Dictamen Final de Infracción (Más de cero elementos ausentes en EPP) 
                # para dicho Sujeto ID
                if len(lista_faltas_texto) > 0:
                    hay_infraccion = True
                    
                    # Generación y resguardo dinámico del Diccionario Infractor validado
                    estado_infraccion = {
                        'ID': p_id, 'Chaleco': falta_chaleco, 'Casco': falta_casco, 
                        'Lentes': falta_lentes, 'Guantes': falta_guantes
                    }
                    
                    # Consolidación gramatical de falta con join y pintado de texto 
                    # flotante bajo la base de la silueta infractora (Color alerta máxima)
                    texto_faltas = "SIN: " + ", ".join(lista_faltas_texto)
                    cv2.putText(frame, texto_faltas, (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Seguimiento del Cronómetro del hilo principal asíncrono
        tiempo_actual = time.time()
        tiempo_desde_ultimo_reporte = tiempo_actual - ultimo_reporte_tiempo

        # Validación limitadora "Rate Limit" anti-SPAMMING (Para evitar mandar miles de 
        # infracciones en caso de fallos sostenidos leves)
        if tiempo_desde_ultimo_reporte < TIEMPO_ENFRIAMIENTO:
            
            # Render visual amigable que avisa a los operadores del Dashboard que el 
            # software no mandará multas por X Segundos
            tiempo_restante = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_ultimo_reporte)
            cv2.putText(frame, f"SISTEMA EN PAUSA: {tiempo_restante}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Anulación explícita del timer en tracking infractor temporal durante periodos 
            # "graciables" 
            inicio_infraccion_tiempo = None 
            
        else: # Estado ideal nominal: Máquina plenamente operativa a la escucha de violaciones al PPE standard
            
            cv2.putText(frame, "SISTEMA ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # ¿Tenemos un trabajador fuera de norma detectado y listo en las variables 
            # condicionales previas?
            if hay_infraccion:
                
                # ¿Era su primera incidencia o ya arrastraba segundos consecutivos? 
                # (Prevenimos que lo castigue de volada un falso positivo de 1 frame)
                if inicio_infraccion_tiempo is None:
                    inicio_infraccion_tiempo = tiempo_actual
                
                # Validador de continuidad fraccionaria (Segundos puros continuos 
                # infractando frente a la red YOLO)
                segundos_infractando = tiempo_actual - inicio_infraccion_tiempo
                
                # ¿El trabajador acumula suficientes segundos sostenidos demostrando 
                # falta de equipo?
                if segundos_infractando >= TIEMPO_CONFIRMACION:
                    
                    # Requerimos el timestamp real formatado para base de datos y 
                    # naming convention de la Foto Evidencia
                    ahora = datetime.now()
                    fecha_str = ahora.strftime("%Y-%m-%d")
                    hora_str = ahora.strftime("%H:%M:%S")
                    
                    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}.jpg"
                    ruta_foto = os.path.join(CARPETA_EVIDENCIAS, nombre_foto)
                    
                    # Interacción File I/O: Guardo la Matriz limpia sin anotaciones al SSD 
                    # como Evidencia Física Oficial JPEG
                    cv2.imwrite(ruta_foto, frame_limpio)
                    
                    # Interacción Stream/Buffer hacia Excel (Guardado en Memoria por 
                    # Añadidura en vez de Sobreescritura a=Append)
                    with open(RUTA_CSV, mode='a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        
                        # Plasmamos una línea entera estructurada conteniendo el Time Series, 
                        # ID Relacional, Flags Bool e Identificador Foto
                        escritor.writerow([
                            fecha_str, hora_str, estado_infraccion['ID'],
                            estado_infraccion['Chaleco'], estado_infraccion['Casco'],
                            estado_infraccion['Lentes'], estado_infraccion['Guantes'], nombre_foto
                        ])
                    
                    # Como hubó inyección comprobada de una Fila a Base de Datos, 
                    # invocamos Re-render forzoso en las UI Tablas Centrales
                    actualizar_dashboard()
                    
                    # Activamos el freno anti-spamming reseteando los variables 
                    # contadores a Fecha de Ahora mismo
                    ultimo_reporte_tiempo = tiempo_actual
                    inicio_infraccion_tiempo = None
                else:
                    # En caso de no superar el test de Segundos de Vida infractora continuos, 
                    # se muestra warning "Confirmando Falta..."
                    cv2.putText(frame, f"CONFIRMANDO FALTA... {int(TIEMPO_CONFIRMACION - segundos_infractando)}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # El Frame estaba limpio, se anula la persecución temporal inmediatamente y 
                # salvamos del castigo al trabajador
                inicio_infraccion_tiempo = None

        # Condifigurador C-level: Open CV trabaja sobre Azul, Verde, Rojo matemáticamente...
        # los Navegadores usan RGB estándar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Disparo Reactivo de Streamlit sobreescribiendo el componente de puerto marco_video 
        # reservado exprofeso 
        marco_video.image(frame_rgb, channels="RGB", use_column_width=True)

        # Condicional limitador temporal exclusivo de Archivos de Video para forzar 
        # velocidad Humana "Playback Ratio Real 1x"
        if fuente_video == "Video de Prueba (MP4)":
            time.sleep(0.03)

    # Bloque finally equivalente: Cuando el ciclo iterador detenga el renderizado 
    # (Usuario desmarcó checkBox de Streamlit), entonces Cerramos Conexón Hard.
    cap.release()
else:
    # Condición ociosa primaria si no interactuan todavía con los checkboxes directos 
    # para lanzar motor neuronal / de inferencia
    st.info("Haz clic en la casilla de arriba para encender la cámara y procesar el video.")