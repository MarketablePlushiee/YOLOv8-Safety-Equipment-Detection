import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Configuración de página
st.set_page_config(
    page_title="SafeVR - Detector & Monitor CCTV de EPP",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeVR - Sistema de Monitoreo & Detección de EPP")
st.markdown("Plataforma de visión por computadora para auditoría de Equipos de Protección Personal (EPP) en entornos industriales.")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo `best.pt`: {e}")
    st.stop()

tab1, tab2 = st.tabs(["📸 Análisis de Imagen Estática", "🎥 Monitoreo CCTV / Video"])

# ---------------------------------------------------------
# PESTAÑA 1: IMAGEN ESTÁTICA
# ---------------------------------------------------------
with tab1:
    st.subheader("Análisis de Fotografía Industrial")
    
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        conf_threshold = st.slider("Sensibilidad (Umbral)", 0.20, 0.80, 0.30, 0.05)
        iou_threshold = st.slider("Solapamiento (IoU)", 0.10, 0.80, 0.40, 0.05)
        uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"], key="img_up")

    with col_view:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Inferencia a resolución de 1024px para rescatar detalles
            results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, imgsz=1024)
            
            # Obtener imagen renderizada y convertir correctamente BGR -> RGB
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, caption="Detección de EPP", use_column_width=True)
            
            # Métricas
            boxes = results[0].boxes
            if len(boxes) > 0:
                counts = {}
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    counts[class_name] = counts.get(class_name, 0) + 1
                
                st.markdown("#### Objetos Detectados")
                cols = st.columns(max(len(counts), 1))
                for idx, (item, count) in enumerate(counts.items()):
                    cols[idx].metric(label=item.upper(), value=count)

# ---------------------------------------------------------
# PESTAÑA 2: VIDEO / CCTV
# ---------------------------------------------------------
with tab2:
    st.subheader("Feed CCTV / Procesamiento de Video")
    video_file = st.file_uploader("Sube un video (MP4, AVI)...", type=["mp4", "avi", "mov"], key="vid_up")
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        col_v1, col_v2 = st.columns([2, 1])
        
        # Elementos fijos de UI para evitar multiplicación de componentes
        st_frame = col_v1.empty()
        st_status = col_v2.empty()
        
        frame_count = 0
        skip_frames = 3  # Procesar 1 de cada 3 cuadros para mayor fluidez
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            
            # Inferencia sobre fotograma en resolución ligera
            results = model.predict(source=frame, conf=0.30, iou=0.40, imgsz=640, verbose=False)
            annotated_frame = results[0].plot()
            
            # Corrección del espacio de color BGR -> RGB
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_rgb, channels="RGB", use_column_width=True)
            
            # Conteo de detecciones
            boxes = results[0].boxes
            counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                counts[class_name] = counts.get(class_name, 0) + 1
            
            # Renderizar estado en contenedor fijo
            with st_status.container():
                st.markdown("### Estado en Vivo")
                st.write(f"👷 Cascos: **{counts.get('helmet', 0)}**")
                st.write(f"🦺 Chalecos: **{counts.get('vest', 0)}**")
                st.write(f"🥾 Botas: **{counts.get('boots', 0)}**")
                st.write(f"🥽 Gafas: **{counts.get('goggles', 0)}**")
                
                if "no-helmet" in counts or "no-vest" in counts:
                    st.error("🚨 Infracción de EPP Detectada")
                else:
                    st.success("✅ EPP en Norma")
                    
        cap.release()
