import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time

# Configuración de página
st.set_page_config(
    page_title="SafeVR - Detector & Monitor CCTV de EPP",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeVR - Sistema de Monitoreo & Detección de EPP")
st.markdown("Plataforma de visión por computadora para auditoría de Equipos de Protección Personal (EPP) en entornos industriales.")

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo `best.pt`: {e}")
    st.stop()

# Navegación por pestañas
tab1, tab2 = st.tabs(["📸 Análisis de Imagen Estática", "🎥 Monitoreo CCTV / Video en Tiempo Real"])

# ---------------------------------------------------------
# PESTAÑA 1: IMAGEN ESTÁTICA
# ---------------------------------------------------------
with tab1:
    st.subheader("Análisis de Fotografía Industrial")
    
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("### Parámetros de Inferencia")
        conf_threshold = st.slider("Sensibilidad (Umbral de confianza)", 0.25, 0.90, 0.35, 0.05, 
                                   help="Un umbral de 0.35 a 0.45 filtra falsos positivos y mantiene solo detecciones sólidas.")
        iou_threshold = st.slider("Solapamiento (IoU)", 0.10, 0.80, 0.45, 0.05,
                                  help="Filtra cajas redundantes o solapadas sobre el mismo objeto.")
        
        uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"], key="img_uploader")

    with col_view:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Inferencia a alta resolución (1280px) para rescatar detalles finos (botas, gafas)
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=1280
            )
            
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detección de EPP Filtrada y Optimizada", use_column_width=True)
            
            # Conteo de objetos detectados
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.markdown("#### Resumen de Objetos Detectados en Foto")
                counts = {}
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    counts[class_name] = counts.get(class_name, 0) + 1
                
                cols = st.columns(len(counts))
                for idx, (item, count) in enumerate(counts.items()):
                    cols[idx].metric(label=item.upper(), value=count)
            else:
                st.info("No se detectaron elementos con el umbral seleccionado.")
        else:
            st.info("Por favor, sube una imagen industrial para iniciar la inspección.")

# ---------------------------------------------------------
# PESTAÑA 2: VIDEO / SIMULACIÓN CCTV
# ---------------------------------------------------------
with tab2:
    st.subheader("Feed de Cámara de Seguridad (CCTV) / Clip Industrial")
    st.markdown("Procesa videos en tiempo real analizando fotograma a fotograma el cumplimiento de normas SST.")
    
    video_file = st.file_uploader("Sube un video de inspección (MP4, AVI, MOV)...", type=["mp4", "avi", "mov"], key="vid_uploader")
    
    col_v1, col_v2 = st.columns([2, 1])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = col_v1.empty()
        st_metrics = col_v2.container()
        
        stop_btn = col_v2.button("Detener Procesamiento")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detección sobre fotograma en tiempo real
            results = model.predict(source=frame, conf=0.35, iou=0.45, imgsz=640)
            annotated_frame = results[0].plot()
            
            # Convertir BGR de OpenCV a RGB para Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
            
            # Métricas del fotograma actual
            boxes = results[0].boxes
            counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                counts[class_name] = counts.get(class_name, 0) + 1
            
            with st_metrics:
                st.markdown("### Estado CCTV en Vivo")
                st.metric("Cascos (`helmet`)", counts.get("helmet", 0))
                st.metric("Chalecos (`vest`)", counts.get("vest", 0))
                st.metric("Botas (`boots`)", counts.get("boots", 0))
                st.metric("Gafas (`goggles`)", counts.get("goggles", 0))
                
                # Alerta visual de infracción
                if "no-helmet" in counts or "no-vest" in counts:
                    st.error("🚨 ALERTA SST: ¡Infracción de EPP detectada en el Feed!")
                else:
                    st.success("✅ Cumplimiento de EPP dentro de norma")
        
        cap.release()
