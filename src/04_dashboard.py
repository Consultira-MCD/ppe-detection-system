"""
ARGUS VISION — Dashboard Operativo HSE (Health, Safety & Environment)
Sistema de monitoreo en tiempo real de Equipos de Protección Personal (EPP)
utilizando YOLOv8 + OpenCV. Detecta: Casco, Chaleco y Lentes de seguridad.
Registra infracciones con evidencia fotográfica y cooldown anti-spam de 30s.
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
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="ARGUS VISION | HSE Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS GLOBAL CORPORATIVO ---
st.markdown("""
<style>
    /* Oculta elementos de Streamlit para look limpio */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Scrollbar oscuro para tablas */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #FF4B4B; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF6B6B; }

    /* Ajuste sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #14141f 100%);
        border-right: 1px solid #2e2e3e;
    }

    /* Elimina margin extra de st.checkbox */
    [data-testid="stCheckbox"] { margin-bottom: 6px; }

    /* Botón primario */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #FF4B4B, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #FF6B6B, #e74c3c) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. VARIABLES DE CONFIGURACIÓN
# ==========================================
MODEL_PATH = "models/yolov8_epp_v2/weights/best.pt"
WEBCAM_INDEX = 0
TIEMPO_CONFIRMACION = 2.0
TIEMPO_ENFRIAMIENTO = 30.0

CARPETA_EVIDENCIAS = "evidencias"
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)
RUTA_CSV = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")

# Clases de detección activas (Guantes y Zapatos: desactivados en v2.0)
CONFIGURACION_VISUAL = {
    'head_helmet':  ("CON CASCO",       (0, 220, 0)),
    'head_nohelmet':("SIN CASCO",       (0, 0, 255)),
    'face_mask':    ("CON MASCARILLA",  (0, 220, 0)),
    'face_nomask':  ("SIN MASCARILLA",  (0, 0, 255)),
    'vest':         ("CON CHALECO",     (0, 220, 0)),
    'glasses':      ("CON LENTES",      (0, 220, 0)),
    'person':       ("PERSONA",         (0, 220, 220))
}

# ==========================================
# 3. FUNCIONES DEL SISTEMA (INTOCABLES)
# ==========================================
@st.cache_resource
def cargar_modelo():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return YOLO(MODEL_PATH), device

def tiene_equipo(px1, py1, px2, py2, lista_coordenadas_equipo):
    """
    Verifica geométricamente si un equipo EPP se encuentra
    dentro del bounding box de la persona detectada.
    """
    for (cx1, cy1, cx2, cy2) in lista_coordenadas_equipo:
        centro_x = (cx1 + cx2) // 2
        centro_y = (cy1 + cy2) // 2
        if px1 < centro_x < px2 and py1 < centro_y < py2:
            return True
    return False

# Carga del modelo al iniciar (cache)
modelo, device = cargar_modelo()
nombres_clases = modelo.names

# ==========================================
# 4. SESSION STATE
# ==========================================
if 'confirm_reset' not in st.session_state:
    st.session_state['confirm_reset'] = False

# ==========================================
# 5. HEADER CORPORATIVO
# ==========================================
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 60%, #0e1117 100%);
    border: 1px solid #2e2e3e;
    border-left: 4px solid #FF4B4B;
    border-radius: 10px;
    padding: 14px 24px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="display: flex; align-items: center; gap: 16px;">
        <span style="font-size: 30px;">🛡️</span>
        <div>
            <div style="font-size: 22px; font-weight: 900; color: #ffffff; letter-spacing: 3px; line-height: 1.1;">
                ARGUS VISION
            </div>
            <div style="font-size: 10px; color: #FF4B4B; letter-spacing: 4px; font-weight: 600; margin-top: 2px;">
                HSE MONITORING SYSTEM — v2.0
            </div>
        </div>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 11px; color: #606070; letter-spacing: 1px; margin-bottom: 4px;">EPP ACTIVO</div>
        <div style="display: flex; gap: 8px;">
            <span style="background:#FF4B4B22; color:#FF4B4B; border:1px solid #FF4B4B55; border-radius:4px; padding:2px 10px; font-size:10px; font-weight:700; letter-spacing:1px;">⛑ CASCO</span>
            <span style="background:#FF8C0022; color:#FF8C00; border:1px solid #FF8C0055; border-radius:4px; padding:2px 10px; font-size:10px; font-weight:700; letter-spacing:1px;">🦺 CHALECO</span>
            <span style="background:#4fc3f722; color:#4fc3f7; border:1px solid #4fc3f755; border-radius:4px; padding:2px 10px; font-size:10px; font-weight:700; letter-spacing:1px;">🥽 LENTES</span>
        </div>
    </div>
    <div style="text-align: right; font-size: 11px; color: #404050;">
        <div>Motor: <span style="color:#69f0ae;">YOLOv8</span></div>
        <div>Dispositivo: <span style="color:#69f0ae;">{device.upper()}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR — PANEL DE CONTROL
# ==========================================
st.sidebar.markdown("""
<div style="
    text-align: center;
    padding: 10px 0 18px 0;
    border-bottom: 1px solid #2e2e3e;
    margin-bottom: 20px;
">
    <div style="font-size: 11px; color: #FF4B4B; letter-spacing: 3px; font-weight: 700;">⚙ PANEL DE CONTROL</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size: 11px; color: #a0a0b0; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;">
    Número de Cámaras
</div>
""", unsafe_allow_html=True)
num_camaras = st.sidebar.radio(
    "", [1, 2], horizontal=True, label_visibility="collapsed"
)

# ---- CÁMARA 1 ----
st.sidebar.markdown("""
<div style="font-size: 10px; color: #FF4B4B; letter-spacing: 2px; font-weight: 700;
            margin-top: 10px; margin-bottom: 6px;">📷 CÁMARA 1</div>
""", unsafe_allow_html=True)
tipo_cam1 = st.sidebar.radio(
    "", ["Webcam", "Video MP4"], key="tipo_cam1", label_visibility="collapsed"
)
archivo_video1 = None
idx_cam1 = 0
if tipo_cam1 == "Webcam":
    idx_cam1 = st.sidebar.number_input(
        "Índice de webcam", min_value=0, max_value=10, value=0, key="idx_cam1"
    )
else:
    archivo_video1 = st.sidebar.file_uploader(
        "Video Cámara 1", type=['mp4', 'mov', 'avi'], key="file_cam1"
    )

# ---- CÁMARA 2 (opcional) ----
archivo_video2 = None
idx_cam2 = 1
tipo_cam2 = "Webcam"
if num_camaras == 2:
    st.sidebar.markdown("""
<div style="font-size: 10px; color: #4fc3f7; letter-spacing: 2px; font-weight: 700;
            margin-top: 10px; margin-bottom: 6px;">📷 CÁMARA 2</div>
""", unsafe_allow_html=True)
    tipo_cam2 = st.sidebar.radio(
        "", ["Webcam", "Video MP4"], key="tipo_cam2", label_visibility="collapsed"
    )
    if tipo_cam2 == "Webcam":
        idx_cam2 = st.sidebar.number_input(
            "Índice de webcam", min_value=0, max_value=10, value=1, key="idx_cam2"
        )
    else:
        archivo_video2 = st.sidebar.file_uploader(
            "Video Cámara 2", type=['mp4', 'mov', 'avi'], key="file_cam2"
        )

st.sidebar.markdown("""
<div style="border-top: 1px solid #2e2e3e; margin: 18px 0;"></div>
<div style="font-size: 11px; color: #a0a0b0; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px;">
    Administración
</div>
""", unsafe_allow_html=True)

# Botón: Descargar Reporte CSV
if os.path.exists(RUTA_CSV):
    with open(RUTA_CSV, "rb") as f:
        csv_bytes = f.read()
    st.sidebar.download_button(
        label="⬇️  Descargar Reporte CSV",
        data=csv_bytes,
        file_name=f"argus_reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.markdown("""
    <div style="
        background: #1a1a2e;
        border: 1px solid #2e2e3e;
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        color: #404050;
        font-size: 12px;
    ">⬇️  Sin reporte disponible</div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

# Botón: Resetear Base de Datos (con confirmación)
if st.sidebar.button("🗑️  Resetear Base de Datos", use_container_width=True):
    st.session_state['confirm_reset'] = True

if st.session_state['confirm_reset']:
    st.sidebar.markdown("""
    <div style="
        background: #2e1515;
        border: 1px solid #FF4B4B55;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 11px;
        color: #FF8888;
        margin-top: 6px;
        line-height: 1.5;
    ">
        ⚠️ Esta acción eliminará <strong>todas</strong> las incidencias registradas y no puede deshacerse.
    </div>
    """, unsafe_allow_html=True)
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("✅ Confirmar", use_container_width=True, key="btn_confirm_reset"):
            if os.path.exists(RUTA_CSV):
                os.remove(RUTA_CSV)
            st.session_state['confirm_reset'] = False
            st.rerun()
    with col_s2:
        if st.button("✖ Cancelar", use_container_width=True, key="btn_cancel_reset"):
            st.session_state['confirm_reset'] = False
            st.rerun()

st.sidebar.markdown(f"""
<div style="
    border-top: 1px solid #2e2e3e;
    margin-top: 24px;
    padding-top: 16px;
    text-align: center;
    font-size: 10px;
    color: #404050;
    line-height: 1.8;
    letter-spacing: 0.5px;
">
    ARGUS VISION &copy; 2025<br>
    YOLOv8 · Streamlit · OpenCV<br>
    <span style="color:#FF4B4B;">HSE Monitoring Platform</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. LAYOUT PRINCIPAL — DOS COLUMNAS
# ==========================================
col_video, col_kpis = st.columns([3, 2], gap="large")

_PLACEHOLDER_SIN_SENIAL = """
<div style="
    background: linear-gradient(135deg, #0e1117, #1a1a2e);
    border: 2px dashed #2a2a3e;
    border-radius: 12px;
    height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #303040;
    gap: 10px;
">
    <div style="font-size: 42px; opacity: 0.4;">📷</div>
    <div style="font-size: 11px; letter-spacing: 3px; color: #505060; font-weight: 700;">{label}</div>
    <div style="font-size: 10px; color: #303040; letter-spacing: 1px;">SIN SEÑAL</div>
</div>
"""

# --- COLUMNA IZQUIERDA: VIDEO ---
with col_video:
    st.markdown("""
    <div style="
        font-size: 10px;
        font-weight: 700;
        color: #606070;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    ">📹 Transmisión de Seguridad</div>
    """, unsafe_allow_html=True)

    run_camera = st.checkbox("🟢  Activar Monitoreo en Vivo", value=False)

    if num_camaras == 1:
        marco_cam1 = st.empty()
        marco_cam2 = None
        if not run_camera:
            marco_cam1.markdown(
                _PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 1"),
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="font-size:10px;color:#FF4B4B;letter-spacing:2px;'
            'font-weight:700;margin-bottom:4px;">📷 CÁMARA 1</div>',
            unsafe_allow_html=True
        )
        marco_cam1 = st.empty()
        st.markdown(
            '<div style="font-size:10px;color:#4fc3f7;letter-spacing:2px;'
            'font-weight:700;margin-top:14px;margin-bottom:4px;">📷 CÁMARA 2</div>',
            unsafe_allow_html=True
        )
        marco_cam2 = st.empty()
        if not run_camera:
            marco_cam1.markdown(
                _PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 1"),
                unsafe_allow_html=True
            )
            marco_cam2.markdown(
                _PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 2"),
                unsafe_allow_html=True
            )

# --- COLUMNA DERECHA: KPIs, GRÁFICA, TABLA ---
with col_kpis:
    st.markdown("""
    <div style="
        font-size: 10px;
        font-weight: 700;
        color: #606070;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    ">📊 Indicadores Operativos</div>
    """, unsafe_allow_html=True)

    contenedor_tarjetas = st.empty()
    contenedor_grafica  = st.empty()
    contenedor_tabla    = st.empty()


# ==========================================
# 8. FUNCIÓN: ACTUALIZAR DASHBOARD
# ==========================================

# Helpers para construir HTML sin indentación de Python
# (evita que Markdown interprete las líneas como bloques de código)
def _kpi_card(color, label, value, sublabel, border_pos='border-top'):
    return (
        '<div style="flex:1;min-width:120px;background:linear-gradient(135deg,#1e1e2e,#252535);'
        f'border:1px solid #2e2e3e;{border_pos}:3px solid {color};border-radius:10px;padding:14px 12px;margin:3px;">'
        f'<div style="font-size:9px;color:#606070;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:900;color:{color};line-height:1;">{value}</div>'
        f'<div style="font-size:9px;color:#505060;margin-top:4px;">{sublabel}</div>'
        '</div>'
    )

def _kpi_card_wide(color, icon, label, value):
    return (
        '<div style="flex:1;min-width:140px;background:linear-gradient(135deg,#1e1e2e,#252535);'
        f'border:1px solid #2e2e3e;border-left:3px solid {color};border-radius:10px;'
        'padding:10px 14px;margin:3px;display:flex;align-items:center;gap:12px;">'
        f'<div style="font-size:20px;">{icon}</div>'
        '<div>'
        f'<div style="font-size:9px;color:#606070;letter-spacing:1.5px;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:14px;font-weight:700;color:{color};font-family:monospace;margin-top:2px;">{value}</div>'
        '</div>'
        '</div>'
    )

def actualizar_dashboard():
    """
    Lee el CSV de incidencias y re-renderiza todos los componentes
    visuales del panel derecho usando HTML/CSS puro.
    NOTA: El HTML se construye en variables Python ANTES de pasarlo a
    st.markdown() para evitar que la indentación del código sea
    interpretada como bloques de código por el parser de Markdown.
    """
    if os.path.exists(RUTA_CSV):
        df = pd.read_csv(
            RUTA_CSV,
            names=['Fecha', 'Hora', 'ID_Persona', 'Chaleco', 'Casco', 'Lentes', 'Nombre_Foto'],
            header=0
        )
        df['Chaleco'] = pd.to_numeric(df['Chaleco'], errors='coerce').fillna(0)
        df['Casco']   = pd.to_numeric(df['Casco'],   errors='coerce').fillna(0)
        df['Lentes']  = pd.to_numeric(df['Lentes'],  errors='coerce').fillna(0)

        if not df.empty:
            total        = len(df)
            ultima_hora  = str(df.iloc[-1]['Hora'])
            ids_unicos   = df['ID_Persona'].nunique()
            hoy          = datetime.now().strftime("%Y-%m-%d")
            hoy_count    = len(df[df['Fecha'] == hoy])
            reincidentes = int((df.groupby('ID_Persona').size() > 1).sum())

            # --------------------------------------------------
            # A) TARJETAS KPI — construidas como variable string
            # --------------------------------------------------
            html_kpis = (
                '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:6px;">'
                + _kpi_card('#FF4B4B', 'Total Infracciones', total,       'acumuladas')
                + _kpi_card('#69f0ae', 'Eventos Hoy',        hoy_count,   'registrados')
                + _kpi_card('#4fc3f7', 'IDs Detectados',     ids_unicos,  'personas únicas')
                + '</div>'
                + '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:6px;">'
                + _kpi_card_wide('#FF8C00', '🕐', 'Último Evento',  ultima_hora)
                + _kpi_card_wide('#e040fb', '🔁', 'Reincidentes',   f'{reincidentes} personas')
                + '</div>'
            )
            with contenedor_tarjetas.container():
                st.markdown(html_kpis, unsafe_allow_html=True)

            # --------------------------------------------------
            # B) GRÁFICA DE BARRAS — construida como variable string
            # --------------------------------------------------
            totales = {
                'Sin Casco':   int(df['Casco'].sum()),
                'Sin Chaleco': int(df['Chaleco'].sum()),
                'Sin Lentes':  int(df['Lentes'].sum()),
            }
            iconos   = ['⛑', '🦺', '🥽']
            colores_barra = ['#FF4B4B', '#FF8C00', '#4fc3f7']
            max_val  = max(totales.values()) if max(totales.values()) > 0 else 1

            html_barras = (
                '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);'
                'border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;margin-bottom:6px;">'
                '<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;">'
                'Infracciones por Tipo de EPP</div>'
            )
            for i, (etiqueta, cantidad) in enumerate(totales.items()):
                pct   = int((cantidad / max_val) * 100)
                color = colores_barra[i]
                icono = iconos[i]
                html_barras += (
                    '<div style="margin-bottom:14px;">'
                    '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">'
                    f'<span style="font-size:12px;color:#c0c0d0;font-weight:500;">{icono} {etiqueta}</span>'
                    f'<span style="font-size:13px;font-weight:800;color:{color};font-family:monospace;">{cantidad}</span>'
                    '</div>'
                    '<div style="background:#1a1a2e;border-radius:6px;width:100%;height:12px;overflow:hidden;">'
                    f'<div style="background:linear-gradient(90deg,{color}88,{color});width:{pct}%;height:100%;border-radius:6px;"></div>'
                    '</div>'
                    '</div>'
                )
            html_barras += '</div>'

            with contenedor_grafica.container():
                st.markdown(html_barras, unsafe_allow_html=True)

            # --------------------------------------------------
            # C) TABLA DE HISTORIAL — construida como variable string
            # --------------------------------------------------
            def badge_epp(val):
                v = int(val)
                if v == 1:
                    return ('<span style="display:inline-block;background:#FF4B4B22;color:#FF4B4B;'
                            'border:1px solid #FF4B4B55;border-radius:4px;padding:1px 7px;'
                            'font-size:10px;font-weight:800;">FALTA</span>')
                return ('<span style="display:inline-block;background:#69f0ae18;color:#69f0ae;'
                        'border:1px solid #69f0ae44;border-radius:4px;padding:1px 9px;'
                        'font-size:10px;font-weight:800;">OK</span>')

            headers_list = ['Fecha', 'Hora', 'ID', 'Casco', 'Chaleco', 'Lentes', 'Evidencia']
            th_html = "".join([
                f'<th style="padding:9px 10px;background:#1a0a0a;color:#FF4B4B;font-size:9px;'
                f'letter-spacing:2px;font-weight:800;text-transform:uppercase;white-space:nowrap;'
                f'border-bottom:2px solid #FF4B4B44;">{h}</th>'
                for h in headers_list
            ])

            df_invertido = df.iloc[::-1].copy()
            filas_html = ""
            for idx, (_, fila) in enumerate(df_invertido.iterrows()):
                bg = "#1e1e2e" if idx % 2 == 0 else "#1a1a28"
                foto_nombre = str(fila['Nombre_Foto'])
                foto_corta  = foto_nombre[:14] + "…" if len(foto_nombre) > 14 else foto_nombre
                filas_html += (
                    f'<tr style="background:{bg};">'
                    f'<td style="padding:7px 10px;color:#707080;font-size:11px;white-space:nowrap;">{fila["Fecha"]}</td>'
                    f'<td style="padding:7px 10px;color:#d0d0d0;font-family:monospace;font-size:11px;">{fila["Hora"]}</td>'
                    f'<td style="padding:7px 10px;color:#4fc3f7;font-size:11px;font-weight:700;text-align:center;">#{fila["ID_Persona"]}</td>'
                    f'<td style="padding:7px 10px;text-align:center;">{badge_epp(fila["Casco"])}</td>'
                    f'<td style="padding:7px 10px;text-align:center;">{badge_epp(fila["Chaleco"])}</td>'
                    f'<td style="padding:7px 10px;text-align:center;">{badge_epp(fila["Lentes"])}</td>'
                    f'<td style="padding:7px 10px;color:#404050;font-size:10px;font-family:monospace;">{foto_corta}</td>'
                    '</tr>'
                )

            html_tabla = (
                '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);'
                'border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;">'
                f'<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:12px;">'
                f'Historial de Incidencias <span style="color:#404050;font-size:9px;margin-left:8px;">({total} registros)</span></div>'
                '<div style="max-height:230px;overflow-y:auto;border-radius:6px;border:1px solid #2a2a3e;">'
                '<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody style="font-size:11px;">{filas_html}</tbody>'
                '</table></div></div>'
            )
            with contenedor_tabla.container():
                st.markdown(html_tabla, unsafe_allow_html=True)

    else:
        # Estado vacío: sin datos aún
        html_vacio = (
            '<div style="background:linear-gradient(135deg,#1a1a2e,#1e1e30);'
            'border:1px solid #2e2e3e;border-radius:10px;padding:28px 20px;text-align:center;">'
            '<div style="font-size:40px;margin-bottom:10px;opacity:0.4;">📂</div>'
            '<div style="font-size:13px;color:#505060;font-weight:600;letter-spacing:1px;">SIN INCIDENCIAS REGISTRADAS</div>'
            '<div style="font-size:11px;color:#353545;margin-top:6px;">Active el monitoreo para comenzar a generar datos.</div>'
            '</div>'
        )
        with contenedor_tarjetas.container():
            st.markdown(html_vacio, unsafe_allow_html=True)


# Renderizado inicial al cargar la app
actualizar_dashboard()

# ==========================================
# 9. FUNCIONES DE APOYO AL BUCLE
# ==========================================

def _abrir_captura(tipo, archivo_video_obj, idx_webcam, sufijo_temp):
    """Abre y devuelve un cv2.VideoCapture según la fuente seleccionada."""
    if tipo == "Webcam":
        return cv2.VideoCapture(idx_webcam)
    else:
        if archivo_video_obj is None:
            return None
        ruta_temp = os.path.join(CARPETA_EVIDENCIAS, f"temp_video_{sufijo_temp}.mp4")
        with open(ruta_temp, "wb") as f:
            f.write(archivo_video_obj.read())
        return cv2.VideoCapture(ruta_temp)


def _procesar_frame(frame, cam_label, cam_color_bgr):
    """
    Ejecuta inferencia YOLO sobre un frame y dibuja overlays.
    Devuelve (frame_anotado_rgb, frame_limpio, hay_infraccion, estado_infraccion).
    """
    frame_limpio = frame.copy()

    resultados = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
    cajas = resultados[0].boxes

    coordenadas_personas = []
    coordenadas_chalecos, coordenadas_cascos, coordenadas_lentes = [], [], []
    estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0}
    hay_infraccion = False

    if cajas is not None and len(cajas) > 0:
        for caja in cajas:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            clase_id    = int(caja.cls[0])
            nombre_yolo = nombres_clases[clase_id]
            track_id    = int(caja.id[0]) if caja.id is not None else -1

            if nombre_yolo == 'person':       coordenadas_personas.append((x1, y1, x2, y2, track_id))
            elif nombre_yolo == 'vest':        coordenadas_chalecos.append((x1, y1, x2, y2))
            elif nombre_yolo == 'head_helmet': coordenadas_cascos.append((x1, y1, x2, y2))
            elif nombre_yolo == 'glasses':     coordenadas_lentes.append((x1, y1, x2, y2))

            texto_mostrar, color = CONFIGURACION_VISUAL.get(nombre_yolo, (nombre_yolo.upper(), (128, 128, 128)))
            if nombre_yolo == 'person' and track_id != -1:
                texto_mostrar = f"PERSONA #{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (px1, py1, px2, py2, p_id) in coordenadas_personas:
            lista_faltas_texto = []
            falta_chaleco = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos) else 0
            falta_casco   = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos)   else 0
            falta_lentes  = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes)   else 0

            if falta_chaleco: lista_faltas_texto.append("Chaleco")
            if falta_casco:   lista_faltas_texto.append("Casco")
            if falta_lentes:  lista_faltas_texto.append("Lentes")

            if lista_faltas_texto:
                hay_infraccion = True
                estado_infraccion = {
                    'ID': p_id, 'Chaleco': falta_chaleco,
                    'Casco': falta_casco, 'Lentes': falta_lentes
                }
                texto_faltas = "SIN: " + ", ".join(lista_faltas_texto)
                cv2.putText(frame, texto_faltas, (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Etiqueta de cámara en esquina superior derecha
    h, w = frame.shape[:2]
    label_size, _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (w - label_size[0] - 14, 6), (w - 4, 28), (0, 0, 0), -1)
    cv2.putText(frame, cam_label, (w - label_size[0] - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam_color_bgr, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, frame_limpio, hay_infraccion, estado_infraccion


def _registrar_infraccion(frame_limpio, estado_infraccion, sufijo_foto=""):
    """Guarda evidencia fotográfica y escribe fila en el CSV."""
    ahora     = datetime.now()
    fecha_str = ahora.strftime("%Y-%m-%d")
    hora_str  = ahora.strftime("%H:%M:%S")
    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}{sufijo_foto}.jpg"
    ruta_foto   = os.path.join(CARPETA_EVIDENCIAS, nombre_foto)
    cv2.imwrite(ruta_foto, frame_limpio)
    with open(RUTA_CSV, mode='a', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            fecha_str, hora_str, estado_infraccion['ID'],
            estado_infraccion['Chaleco'], estado_infraccion['Casco'],
            estado_infraccion['Lentes'], nombre_foto
        ])


# ==========================================
# 10. BUCLE DE PROCESAMIENTO DE VIDEO
# ==========================================
if run_camera:
    # --- Apertura de capturas ---
    cap1 = _abrir_captura(tipo_cam1, archivo_video1, idx_cam1, "cam1")
    cap2 = None
    if num_camaras == 2:
        cap2 = _abrir_captura(tipo_cam2, archivo_video2, idx_cam2, "cam2")
        if cap2 is None:
            st.warning("Cámara 2: sube un archivo de video o verifica el índice de webcam.")

    if cap1 is None:
        st.warning("Cámara 1: sube un archivo de video o verifica el índice de webcam.")
        st.stop()

    # Cronómetros independientes por cámara
    estado_cams = {
        1: {'ultimo_reporte': 0.0, 'inicio_infraccion': None},
        2: {'ultimo_reporte': 0.0, 'inicio_infraccion': None},
    }

    while run_camera:
        tiempo_actual = time.time()
        exito1, frame1 = cap1.read()
        exito2, frame2 = (cap2.read() if cap2 is not None else (False, None))

        if not exito1:
            st.info("Fin de la transmisión — Cámara 1.")
            break

        # ---- Procesar Cámara 1 ----
        rgb1, limpio1, infraccion1, estado1 = _procesar_frame(
            frame1, "CAM 1", (0, 80, 255)   # rojo-naranja en BGR
        )
        e1 = estado_cams[1]
        tiempo_desde_reporte1 = tiempo_actual - e1['ultimo_reporte']
        if tiempo_desde_reporte1 < TIEMPO_ENFRIAMIENTO:
            restante1 = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_reporte1)
            cv2.putText(frame1, f"PAUSA: {restante1}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            e1['inicio_infraccion'] = None
        else:
            cv2.putText(frame1, "ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if infraccion1:
                if e1['inicio_infraccion'] is None:
                    e1['inicio_infraccion'] = tiempo_actual
                seg_infractando = tiempo_actual - e1['inicio_infraccion']
                if seg_infractando >= TIEMPO_CONFIRMACION:
                    _registrar_infraccion(limpio1, estado1, "_c1")
                    actualizar_dashboard()
                    e1['ultimo_reporte'] = tiempo_actual
                    e1['inicio_infraccion'] = None
                else:
                    cv2.putText(frame1,
                                f"CONFIRMANDO... {int(TIEMPO_CONFIRMACION - seg_infractando)}s",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                e1['inicio_infraccion'] = None

        marco_cam1.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_column_width=True)

        # ---- Procesar Cámara 2 (si existe y tiene frame) ----
        if cap2 is not None and marco_cam2 is not None:
            if exito2 and frame2 is not None:
                rgb2, limpio2, infraccion2, estado2 = _procesar_frame(
                    frame2, "CAM 2", (255, 150, 0)   # azul-cyan en BGR
                )
                e2 = estado_cams[2]
                tiempo_desde_reporte2 = tiempo_actual - e2['ultimo_reporte']
                if tiempo_desde_reporte2 < TIEMPO_ENFRIAMIENTO:
                    restante2 = int(TIEMPO_ENFRIAMIENTO - tiempo_desde_reporte2)
                    cv2.putText(frame2, f"PAUSA: {restante2}s", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    e2['inicio_infraccion'] = None
                else:
                    cv2.putText(frame2, "ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if infraccion2:
                        if e2['inicio_infraccion'] is None:
                            e2['inicio_infraccion'] = tiempo_actual
                        seg_infractando = tiempo_actual - e2['inicio_infraccion']
                        if seg_infractando >= TIEMPO_CONFIRMACION:
                            _registrar_infraccion(limpio2, estado2, "_c2")
                            actualizar_dashboard()
                            e2['ultimo_reporte'] = tiempo_actual
                            e2['inicio_infraccion'] = None
                        else:
                            cv2.putText(frame2,
                                        f"CONFIRMANDO... {int(TIEMPO_CONFIRMACION - seg_infractando)}s",
                                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        e2['inicio_infraccion'] = None
                marco_cam2.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_column_width=True)
            else:
                # Cámara 2 sin frames disponibles
                marco_cam2.markdown(
                    _PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 2 — SIN SEÑAL"),
                    unsafe_allow_html=True
                )

        # Retención natural para videos MP4
        if tipo_cam1 == "Video MP4" or tipo_cam2 == "Video MP4":
            time.sleep(0.03)

    # Liberación de capturas
    cap1.release()
    if cap2 is not None:
        cap2.release()
