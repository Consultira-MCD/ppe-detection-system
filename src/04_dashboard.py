"""
ARGUS VISION — Dashboard Operativo HSE (Health, Safety & Environment)
Sistema de monitoreo en tiempo real de Equipos de Protección Personal (EPP)
utilizando YOLOv8 + OpenCV. Detecta: Casco, Chaleco, Lentes y Mascarilla.
Registra infracciones con evidencia fotográfica y cooldown anti-spam de 30s.
"""

import streamlit as st
import pandas as pd
import altair as alt
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

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 0.8rem; padding-bottom: 1rem; max-width: 100%; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #FF4B4B; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF6B6B; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #14141f 100%);
        border-right: 1px solid #2e2e3e;
    }
    [data-testid="stCheckbox"] { margin-bottom: 6px; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #FF4B4B, #c0392b) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #FF6B6B, #e74c3c) !important;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px; background: #0e1117;
        border-bottom: 1px solid #2e2e3e; padding: 0 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background: transparent; border: none;
        color: #606070; font-size: 12px; font-weight: 600;
        letter-spacing: 1.5px; text-transform: uppercase; padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important; color: #FF4B4B !important;
        border-bottom: 2px solid #FF4B4B !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
MODEL_PATH          = "models/yolov8_epp_v2_produccion/weights/best.pt"
TIEMPO_CONFIRMACION = 2.0
TIEMPO_ENFRIAMIENTO = 30.0
CARPETA_EVIDENCIAS  = "evidencias"
os.makedirs(CARPETA_EVIDENCIAS, exist_ok=True)
RUTA_CSV    = os.path.join(CARPETA_EVIDENCIAS, "reporte_incidencias.csv")
CABECERA_CSV = ['Fecha', 'Hora', 'ID_Persona', 'Chaleco', 'Casco', 'Lentes', 'Mascarilla', 'Nombre_Foto']

CONFIGURACION_VISUAL = {
    'head_helmet':   ("CON CASCO",      (0, 220, 0)),
    'head_nohelmet': ("SIN CASCO",      (0, 0, 255)),
    'face_mask':     ("CON MASCARILLA", (0, 220, 0)),
    'face_nomask':   ("SIN MASCARILLA", (0, 0, 255)),
    'vest':          ("CON CHALECO",    (0, 220, 0)),
    'glasses':       ("CON LENTES",     (0, 220, 0)),
    'person':        ("PERSONA",        (0, 220, 220)),
}

def _inicializar_csv():
    if not os.path.exists(RUTA_CSV):
        with open(RUTA_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(CABECERA_CSV)

_inicializar_csv()

# ==========================================
# 3. MODELO Y DATOS
# ==========================================
@st.cache_resource
def cargar_modelo():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return YOLO(MODEL_PATH), device

def tiene_equipo(px1, py1, px2, py2, lista):
    for (cx1, cy1, cx2, cy2) in lista:
        cx = (cx1 + cx2) // 2
        cy = (cy1 + cy2) // 2
        if px1 < cx < px2 and py1 < cy < py2:
            return True
    return False

def cargar_df():
    if not os.path.exists(RUTA_CSV):
        return pd.DataFrame(columns=CABECERA_CSV)
    try:
        with open(RUTA_CSV, 'r') as f:
            primera = f.readline().strip()
        tiene_header = primera == ','.join(CABECERA_CSV)
        if tiene_header:
            df = pd.read_csv(RUTA_CSV)
        else:
            df = pd.read_csv(RUTA_CSV, header=None, names=CABECERA_CSV)
        df = df[df['Fecha'].astype(str) != 'Fecha']
        if df.empty:
            return pd.DataFrame(columns=CABECERA_CSV)
        for col in ['Chaleco', 'Casco', 'Lentes', 'Mascarilla']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        # Forzar columnas string a numpy object para evitar LargeUtf8 de PyArrow con Altair
        for col in ['Fecha', 'Hora', 'ID_Persona', 'Nombre_Foto']:
            if col in df.columns:
                df[col] = df[col].astype(object).astype(str)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=CABECERA_CSV)

def _datos_demo():
    import random
    random.seed(42)
    from datetime import timedelta
    base = datetime.now()
    filas = []
    for i in range(60):
        dia  = base - timedelta(days=random.randint(0, 29))
        hora = f"{random.randint(7,18):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        epp  = [random.randint(0, 1) for _ in range(4)]
        if sum(epp) == 0:
            epp[random.randint(0, 3)] = 1
        filas.append({
            'Fecha': dia.strftime("%Y-%m-%d"),
            'Hora': hora,
            'ID_Persona': str(random.randint(1, 15)),
            'Chaleco': epp[0], 'Casco': epp[1],
            'Lentes': epp[2], 'Mascarilla': epp[3],
            'Nombre_Foto': f"demo_{i:03d}.jpg",
        })
    df = pd.DataFrame(filas)
    for col in ['Chaleco', 'Casco', 'Lentes', 'Mascarilla']:
        df[col] = df[col].astype(int)
    return df

modelo, device = cargar_modelo()
nombres_clases  = modelo.names

# ==========================================
# 4. SESSION STATE
# ==========================================
for _k, _v in [('confirm_reset', False), ('session_start', None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ==========================================
# 5. HEADER CORPORATIVO
# ==========================================
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0e1117 0%,#1a1a2e 60%,#0e1117 100%);
border:1px solid #2e2e3e;border-left:4px solid #FF4B4B;border-radius:10px;
padding:14px 24px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:30px;">🛡️</span>
        <div>
            <div style="font-size:22px;font-weight:900;color:#ffffff;letter-spacing:3px;line-height:1.1;">ARGUS VISION</div>
            <div style="font-size:10px;color:#FF4B4B;letter-spacing:4px;font-weight:600;margin-top:2px;">HSE MONITORING SYSTEM — v2.0</div>
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:11px;color:#606070;letter-spacing:1px;margin-bottom:4px;">EPP ACTIVO</div>
        <div style="display:flex;gap:8px;">
            <span style="background:#FF4B4B22;color:#FF4B4B;border:1px solid #FF4B4B55;border-radius:4px;padding:2px 10px;font-size:10px;font-weight:700;letter-spacing:1px;">⛑ CASCO</span>
            <span style="background:#FF8C0022;color:#FF8C00;border:1px solid #FF8C0055;border-radius:4px;padding:2px 10px;font-size:10px;font-weight:700;letter-spacing:1px;">🦺 CHALECO</span>
            <span style="background:#4fc3f722;color:#4fc3f7;border:1px solid #4fc3f755;border-radius:4px;padding:2px 10px;font-size:10px;font-weight:700;letter-spacing:1px;">🥽 LENTES</span>
            <span style="background:#e040fb22;color:#e040fb;border:1px solid #e040fb55;border-radius:4px;padding:2px 10px;font-size:10px;font-weight:700;letter-spacing:1px;">😷 MASCARILLA</span>
        </div>
    </div>
    <div style="text-align:right;font-size:11px;color:#404050;">
        <div>Motor: <span style="color:#69f0ae;">YOLOv8</span></div>
        <div>Dispositivo: <span style="color:#69f0ae;">{device.upper()}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div style="text-align:center;padding:10px 0 18px 0;border-bottom:1px solid #2e2e3e;margin-bottom:20px;">
    <div style="font-size:11px;color:#FF4B4B;letter-spacing:3px;font-weight:700;">⚙ PANEL DE CONTROL</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size:11px;color:#a0a0b0;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">Número de Cámaras</div>', unsafe_allow_html=True)
num_camaras = st.sidebar.radio("", [1, 2], horizontal=True, label_visibility="collapsed")

# Cámara 1
st.sidebar.markdown('<div style="font-size:10px;color:#FF4B4B;letter-spacing:2px;font-weight:700;margin-top:10px;margin-bottom:6px;">📷 CÁMARA 1</div>', unsafe_allow_html=True)
tipo_cam1 = st.sidebar.radio("", ["Webcam", "Video MP4", "Stream RTSP"], key="tipo_cam1", label_visibility="collapsed")
archivo_video1, idx_cam1, rtsp_url1 = None, 0, ""
if tipo_cam1 == "Webcam":
    idx_cam1 = st.sidebar.number_input("Índice de webcam", min_value=0, max_value=10, value=0, key="idx_cam1")
elif tipo_cam1 == "Video MP4":
    archivo_video1 = st.sidebar.file_uploader("Video Cámara 1", type=['mp4', 'mov', 'avi'], key="file_cam1")
else:
    rtsp_url1 = st.sidebar.text_input("URL RTSP", placeholder="rtsp://user:pass@ip:554/stream", key="rtsp_cam1")

# Cámara 2
archivo_video2, idx_cam2, tipo_cam2, rtsp_url2 = None, 1, "Webcam", ""
if num_camaras == 2:
    st.sidebar.markdown('<div style="font-size:10px;color:#4fc3f7;letter-spacing:2px;font-weight:700;margin-top:10px;margin-bottom:6px;">📷 CÁMARA 2</div>', unsafe_allow_html=True)
    tipo_cam2 = st.sidebar.radio("", ["Webcam", "Video MP4", "Stream RTSP"], key="tipo_cam2", label_visibility="collapsed")
    if tipo_cam2 == "Webcam":
        idx_cam2 = st.sidebar.number_input("Índice de webcam", min_value=0, max_value=10, value=1, key="idx_cam2")
    elif tipo_cam2 == "Video MP4":
        archivo_video2 = st.sidebar.file_uploader("Video Cámara 2", type=['mp4', 'mov', 'avi'], key="file_cam2")
    else:
        rtsp_url2 = st.sidebar.text_input("URL RTSP", placeholder="rtsp://user:pass@ip:554/stream", key="rtsp_cam2")

st.sidebar.markdown('<div style="border-top:1px solid #2e2e3e;margin:18px 0;"></div><div style="font-size:11px;color:#a0a0b0;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Administración</div>', unsafe_allow_html=True)

if os.path.exists(RUTA_CSV):
    with open(RUTA_CSV, "rb") as f:
        csv_bytes = f.read()
    st.sidebar.download_button(
        label="⬇️  Descargar Reporte CSV",
        data=csv_bytes,
        file_name=f"argus_reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.sidebar.markdown('<div style="background:#1a1a2e;border:1px solid #2e2e3e;border-radius:8px;padding:8px 12px;text-align:center;color:#404050;font-size:12px;">⬇️  Sin reporte disponible</div>', unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
if st.sidebar.button("🗑️  Resetear Base de Datos", use_container_width=True):
    st.session_state['confirm_reset'] = True

if st.session_state['confirm_reset']:
    st.sidebar.markdown('<div style="background:#2e1515;border:1px solid #FF4B4B55;border-radius:8px;padding:10px 12px;font-size:11px;color:#FF8888;margin-top:6px;line-height:1.5;">⚠️ Esta acción eliminará <strong>todas</strong> las incidencias y no puede deshacerse.</div>', unsafe_allow_html=True)
    cs1, cs2 = st.sidebar.columns(2)
    with cs1:
        if st.button("✅ Confirmar", use_container_width=True, key="btn_confirm_reset"):
            if os.path.exists(RUTA_CSV):
                os.remove(RUTA_CSV)
            _inicializar_csv()
            st.session_state['confirm_reset'] = False
            st.rerun()
    with cs2:
        if st.button("✖ Cancelar", use_container_width=True, key="btn_cancel_reset"):
            st.session_state['confirm_reset'] = False
            st.rerun()

st.sidebar.markdown("""
<div style="border-top:1px solid #2e2e3e;margin-top:24px;padding-top:16px;text-align:center;
font-size:10px;color:#404050;line-height:1.8;letter-spacing:0.5px;">
ARGUS VISION &copy; 2025<br>YOLOv8 · Streamlit · OpenCV<br>
<span style="color:#FF4B4B;">HSE Monitoring Platform</span></div>
""", unsafe_allow_html=True)

# ==========================================
# 7. HELPERS HTML
# ==========================================
def _kpi_card(color, label, value, sublabel, border_pos='border-top'):
    return (
        '<div style="flex:1;min-width:120px;background:linear-gradient(135deg,#1e1e2e,#252535);'
        f'border:1px solid #2e2e3e;{border_pos}:3px solid {color};border-radius:10px;padding:14px 12px;margin:3px;">'
        f'<div style="font-size:9px;color:#606070;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:900;color:{color};line-height:1;">{value}</div>'
        f'<div style="font-size:9px;color:#505060;margin-top:4px;">{sublabel}</div></div>'
    )

def _kpi_card_wide(color, icon, label, value):
    return (
        '<div style="flex:1;min-width:140px;background:linear-gradient(135deg,#1e1e2e,#252535);'
        f'border:1px solid #2e2e3e;border-left:3px solid {color};border-radius:10px;'
        'padding:10px 14px;margin:3px;display:flex;align-items:center;gap:12px;">'
        f'<div style="font-size:20px;">{icon}</div><div>'
        f'<div style="font-size:9px;color:#606070;letter-spacing:1.5px;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:14px;font-weight:700;color:{color};font-family:monospace;margin-top:2px;">{value}</div>'
        '</div></div>'
    )

_PLACEHOLDER_SIN_SENIAL = """
<div style="background:linear-gradient(135deg,#0e1117,#1a1a2e);border:2px dashed #2a2a3e;
border-radius:12px;height:300px;display:flex;flex-direction:column;align-items:center;
justify-content:center;color:#303040;gap:10px;">
    <div style="font-size:42px;opacity:0.4;">📷</div>
    <div style="font-size:11px;letter-spacing:3px;color:#505060;font-weight:700;">{label}</div>
    <div style="font-size:10px;color:#303040;letter-spacing:1px;">SIN SEÑAL</div>
</div>"""

# ==========================================
# 8. TABS
# ==========================================
tab_monitor, tab_stats, tab_incidencias = st.tabs([
    "📹  Monitor en Vivo",
    "📊  Estadísticas",
    "📋  Incidencias",
])

# ------------------------------------------
# TAB 1 — MONITOR EN VIVO
# ------------------------------------------
with tab_monitor:
    col_video, col_kpis = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown('<div style="font-size:10px;font-weight:700;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:10px;">📹 Transmisión de Seguridad</div>', unsafe_allow_html=True)
        run_camera = st.checkbox("🟢  Activar Monitoreo en Vivo", value=False)

        if num_camaras == 1:
            marco_cam1 = st.empty()
            marco_cam2 = None
            if not run_camera:
                marco_cam1.markdown(_PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 1"), unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:10px;color:#FF4B4B;letter-spacing:2px;font-weight:700;margin-bottom:4px;">📷 CÁMARA 1</div>', unsafe_allow_html=True)
            marco_cam1 = st.empty()
            st.markdown('<div style="font-size:10px;color:#4fc3f7;letter-spacing:2px;font-weight:700;margin-top:14px;margin-bottom:4px;">📷 CÁMARA 2</div>', unsafe_allow_html=True)
            marco_cam2 = st.empty()
            if not run_camera:
                marco_cam1.markdown(_PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 1"), unsafe_allow_html=True)
                marco_cam2.markdown(_PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 2"), unsafe_allow_html=True)

    with col_kpis:
        st.markdown('<div style="font-size:10px;font-weight:700;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:10px;">📊 Indicadores Operativos</div>', unsafe_allow_html=True)
        contenedor_tarjetas = st.empty()
        contenedor_grafica  = st.empty()
        contenedor_tabla    = st.empty()

# ------------------------------------------
# TAB 2 — ESTADÍSTICAS
# ------------------------------------------
with tab_stats:
    df_stats = cargar_df()
    es_demo = df_stats.empty
    if es_demo:
        df_stats = _datos_demo()
        st.markdown('<div style="background:#1a1f0e;border:1px solid #4a6a1a;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:12px;color:#a0c060;">⚠️ Vista de demostración con datos sintéticos — activa el monitoreo y recarga para ver datos reales.</div>', unsafe_allow_html=True)
    if True:
        # Preparación
        df_stats['Hora_num'] = pd.to_datetime(df_stats['Hora'], format='%H:%M:%S', errors='coerce').dt.hour
        total_infr   = len(df_stats)
        ids_unicos_s = df_stats['ID_Persona'].nunique()
        epp_totales  = {
            'Casco':      int(df_stats['Casco'].sum()),
            'Chaleco':    int(df_stats['Chaleco'].sum()),
            'Lentes':     int(df_stats['Lentes'].sum()),
            'Mascarilla': int(df_stats['Mascarilla'].sum()),
        }
        epp_critico = max(epp_totales, key=epp_totales.get)
        hora_counts  = df_stats['Hora_num'].value_counts()
        hora_pico    = int(hora_counts.idxmax()) if not hora_counts.empty else 0

        # KPIs resumen
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:12px;">Resumen General</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:10px;">'
            + _kpi_card('#FF4B4B', 'Total Infracciones', total_infr,    'acumuladas')
            + _kpi_card('#4fc3f7', 'Trabajadores Únicos', ids_unicos_s, 'con incidencias')
            + _kpi_card('#e040fb', 'EPP más Crítico',     epp_critico,  'mayor incidencia')
            + _kpi_card('#FF8C00', 'Hora Pico',           f'{hora_pico:02d}:00', 'mayor actividad')
            + '</div>',
            unsafe_allow_html=True,
        )

        # Tendencia por día
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-top:16px;margin-bottom:8px;">Infracciones por Día (últimos 30 días)</div>', unsafe_allow_html=True)
        df_dia = df_stats.groupby('Fecha').size().reset_index(name='Infracciones').tail(30)
        df_dia['Fecha'] = df_dia['Fecha'].astype(str).astype(object)
        df_dia['Infracciones'] = df_dia['Infracciones'].astype(int)
        chart_dia = (
            alt.Chart(df_dia)
            .mark_bar(color='#FF4B4B', cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X('Fecha:N', sort=None, title='Fecha',
                         axis=alt.Axis(labelAngle=-45, labelColor='#a0a0b0', titleColor='#606070')),
                y=alt.Y('Infracciones:Q', title='Infracciones',
                         axis=alt.Axis(labelColor='#a0a0b0', titleColor='#606070', gridColor='#2e2e3e')),
                tooltip=['Fecha:N', 'Infracciones:Q'],
            )
            .properties(height=220, background='#1e1e2e')
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart_dia, use_container_width=True)

        # Top infractores + distribución horaria
        col_top, col_hora = st.columns(2, gap="medium")

        with col_top:
            st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;">Top 5 Infractores</div>', unsafe_allow_html=True)
            top5 = (
                df_stats.groupby('ID_Persona').size()
                .sort_values(ascending=False).head(5)
                .reset_index(name='Eventos')
            )
            top5['ID_Persona'] = top5['ID_Persona'].astype(str).astype(object)
            top5['Eventos'] = top5['Eventos'].astype(int)
            chart_top = (
                alt.Chart(top5)
                .mark_bar(color='#e040fb', cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    y=alt.Y('ID_Persona:N', sort='-x', title='ID Persona',
                             axis=alt.Axis(labelColor='#a0a0b0', titleColor='#606070')),
                    x=alt.X('Eventos:Q', title='Infracciones',
                             axis=alt.Axis(labelColor='#a0a0b0', titleColor='#606070', gridColor='#2e2e3e')),
                    tooltip=['ID_Persona:N', 'Eventos:Q'],
                )
                .properties(height=200, background='#1e1e2e')
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_top, use_container_width=True)

        with col_hora:
            st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;">Distribución por Hora</div>', unsafe_allow_html=True)
            df_hora_g = df_stats.groupby('Hora_num').size().reset_index(name='Infracciones')
            df_hora_g['Hora_label'] = df_hora_g['Hora_num'].apply(lambda h: f'{int(h):02d}:00').astype(object)
            df_hora_g['Infracciones'] = df_hora_g['Infracciones'].astype(int)
            chart_hora = (
                alt.Chart(df_hora_g)
                .mark_bar(color='#4fc3f7', cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X('Hora_label:N', sort=None, title='Hora',
                             axis=alt.Axis(labelAngle=-45, labelColor='#a0a0b0', titleColor='#606070')),
                    y=alt.Y('Infracciones:Q', title='',
                             axis=alt.Axis(labelColor='#a0a0b0', gridColor='#2e2e3e')),
                    tooltip=['Hora_label:N', 'Infracciones:Q'],
                )
                .properties(height=200, background='#1e1e2e')
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_hora, use_container_width=True)

        # Desglose EPP (barras HTML)
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-top:8px;margin-bottom:12px;">Desglose por Tipo de EPP</div>', unsafe_allow_html=True)
        _iconos_epp  = ['⛑', '🦺', '🥽', '😷']
        _colores_epp = ['#FF4B4B', '#FF8C00', '#4fc3f7', '#e040fb']
        _etiq_epp    = ['Casco', 'Chaleco', 'Lentes', 'Mascarilla']
        max_epp      = max(epp_totales.values()) if max(epp_totales.values()) > 0 else 1
        html_epp = '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;">'
        for i, etq in enumerate(_etiq_epp):
            val   = epp_totales[etq]
            pct   = int((val / max_epp) * 100)
            color = _colores_epp[i]
            icono = _iconos_epp[i]
            pct_t = f"{(val/total_infr*100):.1f}%" if total_infr > 0 else "0%"
            html_epp += (
                '<div style="margin-bottom:14px;">'
                '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">'
                f'<span style="font-size:12px;color:#c0c0d0;font-weight:500;">{icono} Sin {etq}</span>'
                f'<span style="font-size:12px;font-weight:800;color:{color};font-family:monospace;">'
                f'{val} <span style="font-size:10px;color:#606070;">({pct_t})</span></span></div>'
                '<div style="background:#1a1a2e;border-radius:6px;width:100%;height:12px;overflow:hidden;">'
                f'<div style="background:linear-gradient(90deg,{color}88,{color});width:{pct}%;height:100%;border-radius:6px;"></div>'
                '</div></div>'
            )
        html_epp += '</div>'
        st.markdown(html_epp, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3 — INCIDENCIAS
# ------------------------------------------
with tab_incidencias:
    df_inc = cargar_df()

    if df_inc.empty:
        st.markdown('<div style="background:linear-gradient(135deg,#1a1a2e,#1e1e30);border:1px solid #2e2e3e;border-radius:10px;padding:40px 20px;text-align:center;margin-top:20px;"><div style="font-size:40px;margin-bottom:10px;">📋</div><div style="font-size:14px;color:#c0c0d0;font-weight:600;letter-spacing:1px;">SIN INCIDENCIAS REGISTRADAS</div><div style="font-size:12px;color:#8080a0;margin-top:8px;">Active el monitoreo, genera incidencias y recarga la página para verlas aquí.</div></div>', unsafe_allow_html=True)
    else:
        # Filtros
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:12px;">Filtros</div>', unsafe_allow_html=True)
        df_inc['Fecha_dt'] = pd.to_datetime(df_inc['Fecha'], errors='coerce')
        fecha_min = df_inc['Fecha_dt'].min().date()
        fecha_max = df_inc['Fecha_dt'].max().date()

        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            rango_fechas = st.date_input("Rango de fechas", value=(fecha_min, fecha_max),
                                         min_value=fecha_min, max_value=fecha_max, key="filtro_fechas")
        with col_f2:
            epp_sel = st.multiselect("EPP faltante", options=['Casco', 'Chaleco', 'Lentes', 'Mascarilla'],
                                     default=[], key="filtro_epp")

        # Aplicar filtros
        df_f = df_inc.copy()
        if isinstance(rango_fechas, (list, tuple)) and len(rango_fechas) == 2:
            df_f = df_f[(df_f['Fecha_dt'].dt.date >= rango_fechas[0]) & (df_f['Fecha_dt'].dt.date <= rango_fechas[1])]
        for epp in epp_sel:
            df_f = df_f[df_f[epp] == 1]

        # Métricas del filtro
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Registros encontrados", len(df_f))
        cm2.metric("Personas únicas", df_f['ID_Persona'].nunique())
        cm3.metric("Total en base de datos", len(df_inc))

        # Descarga filtrada
        if not df_f.empty:
            csv_exp = df_f.drop(columns=['Fecha_dt'], errors='ignore').to_csv(index=False).encode('utf-8')
            st.download_button("⬇️  Exportar selección como CSV", data=csv_exp,
                               file_name=f"argus_filtrado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")

        # Tabla HTML — evita st.dataframe() y su serialización Arrow (LargeUtf8)
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-top:16px;margin-bottom:8px;">Registro de Incidencias</div>', unsafe_allow_html=True)
        df_mostrar = df_f.drop(columns=['Fecha_dt'], errors='ignore').iloc[::-1]
        total_f = len(df_mostrar)

        def _badge(val):
            v = int(val)
            if v == 1:
                return '<span style="display:inline-block;background:#FF4B4B22;color:#FF4B4B;border:1px solid #FF4B4B55;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:800;">FALTA</span>'
            return '<span style="display:inline-block;background:#69f0ae18;color:#69f0ae;border:1px solid #69f0ae44;border-radius:4px;padding:1px 9px;font-size:10px;font-weight:800;">OK</span>'

        headers = ['Fecha', 'Hora', 'ID', 'Casco', 'Chaleco', 'Lentes', 'Mascarilla', 'Evidencia']
        th = ''.join(
            f'<th style="padding:9px 10px;background:#1a0a0a;color:#FF4B4B;font-size:9px;'
            f'letter-spacing:2px;font-weight:800;text-transform:uppercase;white-space:nowrap;'
            f'border-bottom:2px solid #FF4B4B44;">{h}</th>'
            for h in headers
        )
        filas = ''
        for i, (_, r) in enumerate(df_mostrar.iterrows()):
            bg = '#1e1e2e' if i % 2 == 0 else '#1a1a28'
            foto = str(r['Nombre_Foto'])
            foto_c = foto[:16] + '…' if len(foto) > 16 else foto
            filas += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:7px 10px;color:#707080;font-size:11px;white-space:nowrap;">{r["Fecha"]}</td>'
                f'<td style="padding:7px 10px;color:#d0d0d0;font-family:monospace;font-size:11px;">{r["Hora"]}</td>'
                f'<td style="padding:7px 10px;color:#4fc3f7;font-size:11px;font-weight:700;text-align:center;">#{r["ID_Persona"]}</td>'
                f'<td style="padding:7px 10px;text-align:center;">{_badge(r["Casco"])}</td>'
                f'<td style="padding:7px 10px;text-align:center;">{_badge(r["Chaleco"])}</td>'
                f'<td style="padding:7px 10px;text-align:center;">{_badge(r["Lentes"])}</td>'
                f'<td style="padding:7px 10px;text-align:center;">{_badge(r["Mascarilla"])}</td>'
                f'<td style="padding:7px 10px;color:#404050;font-size:10px;font-family:monospace;">{foto_c}</td>'
                '</tr>'
            )

        st.markdown(
            '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;">'
            f'<div style="font-size:10px;color:#606070;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">'
            f'Incidencias filtradas <span style="color:#404050;font-size:9px;margin-left:8px;">({total_f} registros)</span></div>'
            '<div style="max-height:400px;overflow-y:auto;border-radius:6px;border:1px solid #2a2a3e;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr>{th}</tr></thead>'
            f'<tbody style="font-size:11px;">{filas}</tbody>'
            '</table></div></div>',
            unsafe_allow_html=True,
        )

        # Visor de evidencia
        st.markdown('<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-top:20px;margin-bottom:8px;">Visor de Evidencia Fotográfica</div>', unsafe_allow_html=True)
        fotos_ok = [
            f for f in df_mostrar['Nombre_Foto'].dropna().tolist()
            if os.path.exists(os.path.join(CARPETA_EVIDENCIAS, str(f)))
        ]
        if fotos_ok:
            foto_sel = st.selectbox("Foto", options=fotos_ok,
                                    format_func=lambda x: f"📸 {x}",
                                    label_visibility="collapsed")
            col_img, col_meta = st.columns([2, 1])
            with col_img:
                st.image(os.path.join(CARPETA_EVIDENCIAS, foto_sel),
                         caption=foto_sel, use_column_width=True)
            with col_meta:
                fila_foto = df_mostrar[df_mostrar['Nombre_Foto'] == foto_sel]
                if not fila_foto.empty:
                    f = fila_foto.iloc[0]
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);border:1px solid #2e2e3e;'
                        'border-radius:10px;padding:16px;">'
                        f'<div style="font-size:9px;color:#606070;letter-spacing:2px;margin-bottom:12px;">DETALLES</div>'
                        f'<div style="margin-bottom:8px;"><span style="font-size:9px;color:#606070;">FECHA</span><br>'
                        f'<span style="color:#c0c0d0;font-size:13px;">{f["Fecha"]}</span></div>'
                        f'<div style="margin-bottom:8px;"><span style="font-size:9px;color:#606070;">HORA</span><br>'
                        f'<span style="color:#c0c0d0;font-family:monospace;font-size:13px;">{f["Hora"]}</span></div>'
                        f'<div style="margin-bottom:12px;"><span style="font-size:9px;color:#606070;">ID PERSONA</span><br>'
                        f'<span style="color:#4fc3f7;font-size:18px;font-weight:700;">#{f["ID_Persona"]}</span></div>'
                        + ''.join([
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                            f'<span style="font-size:11px;color:#a0a0b0;">{epp}</span>'
                            f'<span style="font-size:11px;font-weight:700;color:{"#FF4B4B" if f[epp]==1 else "#69f0ae"};">'
                            f'{"FALTA" if f[epp]==1 else "OK"}</span></div>'
                            for epp in ['Casco', 'Chaleco', 'Lentes', 'Mascarilla']
                        ])
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown('<div style="background:#1a1a2e;border:1px solid #2e2e3e;border-radius:8px;padding:20px;text-align:center;color:#404050;font-size:12px;">Sin imágenes de evidencia disponibles en esta selección.</div>', unsafe_allow_html=True)

# ==========================================
# 9. FUNCIÓN ACTUALIZAR DASHBOARD (MONITOR)
# ==========================================
def actualizar_dashboard():
    df = cargar_df()
    if not df.empty:
        total        = len(df)
        ultima_hora  = str(df.iloc[-1]['Hora'])
        ids_unicos   = df['ID_Persona'].nunique()
        hoy          = datetime.now().strftime("%Y-%m-%d")
        hoy_count    = len(df[df['Fecha'] == hoy])
        reincidentes = int((df.groupby('ID_Persona').size() > 1).sum())

        html_kpis = (
            '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:6px;">'
            + _kpi_card('#FF4B4B', 'Total Infracciones', total,      'acumuladas')
            + _kpi_card('#69f0ae', 'Eventos Hoy',        hoy_count,  'registrados')
            + _kpi_card('#4fc3f7', 'IDs Detectados',     ids_unicos, 'personas únicas')
            + '</div>'
            + '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:6px;">'
            + _kpi_card_wide('#FF8C00', '🕐', 'Último Evento', ultima_hora)
            + _kpi_card_wide('#e040fb', '🔁', 'Reincidentes',  f'{reincidentes} personas')
            + '</div>'
        )
        with contenedor_tarjetas.container():
            st.markdown(html_kpis, unsafe_allow_html=True)

        totales = {
            'Sin Casco':      int(df['Casco'].sum()),
            'Sin Chaleco':    int(df['Chaleco'].sum()),
            'Sin Lentes':     int(df['Lentes'].sum()),
            'Sin Mascarilla': int(df['Mascarilla'].sum()),
        }
        _iconos_b   = ['⛑', '🦺', '🥽', '😷']
        _colores_b  = ['#FF4B4B', '#FF8C00', '#4fc3f7', '#e040fb']
        max_val     = max(totales.values()) if max(totales.values()) > 0 else 1

        html_barras = (
            '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);'
            'border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;margin-bottom:6px;">'
            '<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;">'
            'Infracciones por Tipo de EPP</div>'
        )
        for i, (etiqueta, cantidad) in enumerate(totales.items()):
            pct   = int((cantidad / max_val) * 100)
            color = _colores_b[i]
            icono = _iconos_b[i]
            html_barras += (
                '<div style="margin-bottom:14px;">'
                '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">'
                f'<span style="font-size:12px;color:#c0c0d0;font-weight:500;">{icono} {etiqueta}</span>'
                f'<span style="font-size:13px;font-weight:800;color:{color};font-family:monospace;">{cantidad}</span>'
                '</div>'
                '<div style="background:#1a1a2e;border-radius:6px;width:100%;height:12px;overflow:hidden;">'
                f'<div style="background:linear-gradient(90deg,{color}88,{color});width:{pct}%;height:100%;border-radius:6px;"></div>'
                '</div></div>'
            )
        html_barras += '</div>'
        with contenedor_grafica.container():
            st.markdown(html_barras, unsafe_allow_html=True)

        def badge_epp(val):
            v = int(val)
            if v == 1:
                return '<span style="display:inline-block;background:#FF4B4B22;color:#FF4B4B;border:1px solid #FF4B4B55;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:800;">FALTA</span>'
            return '<span style="display:inline-block;background:#69f0ae18;color:#69f0ae;border:1px solid #69f0ae44;border-radius:4px;padding:1px 9px;font-size:10px;font-weight:800;">OK</span>'

        headers_list = ['Fecha', 'Hora', 'ID', 'Casco', 'Chaleco', 'Lentes', 'Mascarilla', 'Evidencia']
        th_html = "".join([
            f'<th style="padding:9px 10px;background:#1a0a0a;color:#FF4B4B;font-size:9px;letter-spacing:2px;font-weight:800;text-transform:uppercase;white-space:nowrap;border-bottom:2px solid #FF4B4B44;">{h}</th>'
            for h in headers_list
        ])

        filas_html = ""
        for idx, (_, fila) in enumerate(df.iloc[::-1].head(20).iterrows()):
            bg         = "#1e1e2e" if idx % 2 == 0 else "#1a1a28"
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
                f'<td style="padding:7px 10px;text-align:center;">{badge_epp(fila["Mascarilla"])}</td>'
                f'<td style="padding:7px 10px;color:#404050;font-size:10px;font-family:monospace;">{foto_corta}</td>'
                '</tr>'
            )

        html_tabla = (
            '<div style="background:linear-gradient(135deg,#1e1e2e,#252535);border:1px solid #2e2e3e;border-radius:10px;padding:16px 18px;">'
            f'<div style="font-size:10px;color:#606070;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:12px;">Historial de Incidencias <span style="color:#404050;font-size:9px;margin-left:8px;">({total} registros)</span></div>'
            '<div style="max-height:230px;overflow-y:auto;border-radius:6px;border:1px solid #2a2a3e;">'
            f'<table style="width:100%;border-collapse:collapse;"><thead><tr>{th_html}</tr></thead>'
            f'<tbody style="font-size:11px;">{filas_html}</tbody></table></div></div>'
        )
        with contenedor_tabla.container():
            st.markdown(html_tabla, unsafe_allow_html=True)

    else:
        with contenedor_tarjetas.container():
            st.markdown(
                '<div style="background:linear-gradient(135deg,#1a1a2e,#1e1e30);border:1px solid #2e2e3e;border-radius:10px;padding:28px 20px;text-align:center;">'
                '<div style="font-size:40px;margin-bottom:10px;opacity:0.4;">📂</div>'
                '<div style="font-size:13px;color:#505060;font-weight:600;letter-spacing:1px;">SIN INCIDENCIAS REGISTRADAS</div>'
                '<div style="font-size:11px;color:#353545;margin-top:6px;">Active el monitoreo para comenzar a generar datos.</div></div>',
                unsafe_allow_html=True,
            )


actualizar_dashboard()

# ==========================================
# 10. FUNCIONES DEL BUCLE DE VIDEO
# ==========================================
def _abrir_captura(tipo, archivo_video_obj, idx_webcam, rtsp_url, sufijo_temp):
    if tipo == "Webcam":
        return cv2.VideoCapture(idx_webcam)
    if tipo == "Stream RTSP":
        return cv2.VideoCapture(rtsp_url) if rtsp_url else None
    if archivo_video_obj is None:
        return None
    ruta_temp = os.path.join(CARPETA_EVIDENCIAS, f"temp_video_{sufijo_temp}.mp4")
    with open(ruta_temp, "wb") as f:
        f.write(archivo_video_obj.read())
    return cv2.VideoCapture(ruta_temp)


def _procesar_frame(frame, cam_label, cam_color_bgr):
    frame_limpio = frame.copy()
    resultados   = modelo.track(frame, conf=0.6, device=device, persist=True, verbose=False)
    cajas        = resultados[0].boxes

    coordenadas_personas   = []
    coordenadas_chalecos   = []
    coordenadas_cascos     = []
    coordenadas_lentes     = []
    coordenadas_mascarilla = []
    estado_infraccion = {'ID': -1, 'Chaleco': 0, 'Casco': 0, 'Lentes': 0, 'Mascarilla': 0}
    hay_infraccion    = False

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
            elif nombre_yolo == 'face_mask':   coordenadas_mascarilla.append((x1, y1, x2, y2))

            if nombre_yolo not in CONFIGURACION_VISUAL:
                continue
            texto_mostrar, color = CONFIGURACION_VISUAL[nombre_yolo]
            if nombre_yolo == 'person' and track_id != -1:
                texto_mostrar = f"PERSONA #{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, texto_mostrar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (px1, py1, px2, py2, p_id) in coordenadas_personas:
            lista_faltas_texto = []
            falta_chaleco    = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_chalecos)   else 0
            falta_casco      = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_cascos)      else 0
            falta_lentes     = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_lentes)      else 0
            falta_mascarilla = 1 if not tiene_equipo(px1, py1, px2, py2, coordenadas_mascarilla)  else 0

            if falta_chaleco:    lista_faltas_texto.append("Chaleco")
            if falta_casco:      lista_faltas_texto.append("Casco")
            if falta_lentes:     lista_faltas_texto.append("Lentes")
            if falta_mascarilla: lista_faltas_texto.append("Mascarilla")

            if lista_faltas_texto:
                hay_infraccion = True
                estado_infraccion = {
                    'ID': p_id, 'Chaleco': falta_chaleco, 'Casco': falta_casco,
                    'Lentes': falta_lentes, 'Mascarilla': falta_mascarilla,
                }
                cv2.putText(frame, "SIN: " + ", ".join(lista_faltas_texto),
                            (px1 + 5, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    h, w = frame.shape[:2]
    label_size, _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (w - label_size[0] - 14, 6), (w - 4, 28), (0, 0, 0), -1)
    cv2.putText(frame, cam_label, (w - label_size[0] - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam_color_bgr, 2)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_limpio, hay_infraccion, estado_infraccion


def _registrar_infraccion(frame_limpio, estado_infraccion, sufijo_foto=""):
    ahora       = datetime.now()
    fecha_str   = ahora.strftime("%Y-%m-%d")
    hora_str    = ahora.strftime("%H:%M:%S")
    nombre_foto = f"falta_{ahora.strftime('%H%M%S')}{sufijo_foto}.jpg"
    cv2.imwrite(os.path.join(CARPETA_EVIDENCIAS, nombre_foto), frame_limpio)
    _inicializar_csv()
    with open(RUTA_CSV, mode='a', newline='') as archivo:
        csv.writer(archivo).writerow([
            fecha_str, hora_str, estado_infraccion['ID'],
            estado_infraccion['Chaleco'], estado_infraccion['Casco'],
            estado_infraccion['Lentes'], estado_infraccion['Mascarilla'],
            nombre_foto,
        ])


# ==========================================
# 11. BUCLE DE VIDEO
# ==========================================
if run_camera:
    if st.session_state['session_start'] is None:
        st.session_state['session_start'] = datetime.now()

    cap1 = _abrir_captura(tipo_cam1, archivo_video1, idx_cam1, rtsp_url1, "cam1")
    cap2 = None
    if num_camaras == 2:
        cap2 = _abrir_captura(tipo_cam2, archivo_video2, idx_cam2, rtsp_url2, "cam2")
        if cap2 is None:
            st.warning("Cámara 2: sube un archivo de video, verifica el índice o ingresa una URL RTSP válida.")

    if cap1 is None:
        st.warning("Cámara 1: sube un archivo de video, verifica el índice o ingresa una URL RTSP válida.")
        st.stop()

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

        rgb1, limpio1, infraccion1, estado1 = _procesar_frame(frame1, "CAM 1", (0, 80, 255))
        e1 = estado_cams[1]
        tdiff1 = tiempo_actual - e1['ultimo_reporte']
        if tdiff1 < TIEMPO_ENFRIAMIENTO:
            cv2.putText(frame1, f"PAUSA: {int(TIEMPO_ENFRIAMIENTO - tdiff1)}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            e1['inicio_infraccion'] = None
        else:
            cv2.putText(frame1, "ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if infraccion1:
                if e1['inicio_infraccion'] is None:
                    e1['inicio_infraccion'] = tiempo_actual
                seg = tiempo_actual - e1['inicio_infraccion']
                if seg >= TIEMPO_CONFIRMACION:
                    _registrar_infraccion(limpio1, estado1, "_c1")
                    actualizar_dashboard()
                    e1['ultimo_reporte']     = tiempo_actual
                    e1['inicio_infraccion']  = None
                else:
                    cv2.putText(frame1, f"CONFIRMANDO... {int(TIEMPO_CONFIRMACION - seg)}s",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                e1['inicio_infraccion'] = None

        marco_cam1.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        if cap2 is not None and marco_cam2 is not None:
            if exito2 and frame2 is not None:
                rgb2, limpio2, infraccion2, estado2 = _procesar_frame(frame2, "CAM 2", (255, 150, 0))
                e2 = estado_cams[2]
                tdiff2 = tiempo_actual - e2['ultimo_reporte']
                if tdiff2 < TIEMPO_ENFRIAMIENTO:
                    cv2.putText(frame2, f"PAUSA: {int(TIEMPO_ENFRIAMIENTO - tdiff2)}s",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    e2['inicio_infraccion'] = None
                else:
                    cv2.putText(frame2, "ACTIVO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if infraccion2:
                        if e2['inicio_infraccion'] is None:
                            e2['inicio_infraccion'] = tiempo_actual
                        seg = tiempo_actual - e2['inicio_infraccion']
                        if seg >= TIEMPO_CONFIRMACION:
                            _registrar_infraccion(limpio2, estado2, "_c2")
                            actualizar_dashboard()
                            e2['ultimo_reporte']    = tiempo_actual
                            e2['inicio_infraccion'] = None
                        else:
                            cv2.putText(frame2, f"CONFIRMANDO... {int(TIEMPO_CONFIRMACION - seg)}s",
                                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        e2['inicio_infraccion'] = None
                marco_cam2.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            else:
                marco_cam2.markdown(_PLACEHOLDER_SIN_SENIAL.format(label="CÁMARA 2 — SIN SEÑAL"), unsafe_allow_html=True)

        if tipo_cam1 == "Video MP4" or tipo_cam2 == "Video MP4":
            time.sleep(0.03)

    cap1.release()
    if cap2 is not None:
        cap2.release()
    st.session_state['session_start'] = None
