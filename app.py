import streamlit as st
from ultralytics import YOLO
import PIL.Image
import cv2
import numpy as np
import pandas as pd

# Configuración de página
st.set_page_config(page_title="SST Vision - Auditoría de EPP", layout="wide")
st.title("🛡️ SST Vision - Sistema de Auditoría & Monitoreo de EPP")

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Panel de control lateral
st.sidebar.header("⚙️ Configuración de Inferencia")

use_sahi = st.sidebar.checkbox(
    "🔬 Activar Modo Gigapixel SAHI (Fotos de alta resolución)", 
    value=False,
    help="Divide la imagen en recortes de 640x640 para accesorios pequeños a gran distancia."
)

if use_sahi:
    st.sidebar.warning("⚠️ Modo SAHI activo: El procesamiento puede tomar unos segundos adicionales.")

auto_mode = st.sidebar.checkbox("🤖 Sensibilidad Automática Adaptativa (Por Foto)", value=True)

if not auto_mode:
    conf_general_user = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.05, 0.80, 0.15, 0.05)
    conf_small_user = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.05, 0.01)

iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.30, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

uploaded_file = st.file_uploader("Subir fotografía de inspección industrial...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Cargar imagen original en formato RGB puro para la IA
    image = PIL.Image.open(uploaded_file).convert("RGB")
    
    # 2. Inferencia: La IA analiza los colores REALES y NATURALES
    results = model.predict(
        source=image,
        conf=0.03,
        iou=iou_thresh,
        imgsz=1024,
        augment=use_tta,
        agnostic_nms=True
    )
    
    res = results[0]
    boxes = res.boxes
    
    # 3. Lógica de Umbrales Adaptativos
    if auto_mode:
        floors = {'helmet': 0.12, 'vest': 0.08, 'boots': 0.05, 'gloves': 0.04, 'goggles': 0.04}
        max_scores = {k: 0.0 for k in floors.keys()}
        for box in boxes:
            c_name = res.names[int(box.cls[0])]
            score = float(box.conf[0])
            if c_name in max_scores and score > max_scores[c_name]:
                max_scores[c_name] = score
                
        class_thresholds = {}
        for c_name, floor_val in floors.items():
            if max_scores[c_name] > 0:
                class_thresholds[c_name] = max(floor_val, min(max_scores[c_name] * 0.35, 0.20))
            else:
                class_thresholds[c_name] = floor_val
    else:
        class_thresholds = {
            'helmet': conf_general_user, 'vest': conf_general_user,
            'boots': conf_small_user, 'gloves': conf_small_user, 'goggles': conf_small_user
        }

    # 4. Filtrar detecciones
    valid_indices = []
    helmet_boxes = []
    vest_boxes = []
    
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        class_name = res.names[cls_id]
        
        target_thresh = class_thresholds.get(class_name, 0.08)
        if score >= target_thresh:
            valid_indices.append(idx)
            xyxy = box.xyxy[0].cpu().numpy()
            if class_name == 'helmet':
                helmet_boxes.append(xyxy)
            elif class_name == 'vest':
                vest_boxes.append(xyxy)
            
    filtered_res = res[valid_indices]
    
    # 5. Motor de Cumplimiento SST
    compliant_count = 0
    infraction_count = 0
    
    for h_box in helmet_boxes:
        h_x_center = (h_box[0] + h_box[2]) / 2.0
        h_y2 = h_box[3]
        
        has_vest = False
        for v_box in vest_boxes:
            v_x1, v_y1, v_x2, v_y2 = v_box[0], v_box[1], v_box[2], v_box[3]
            if (v_x1 <= h_x_center <= v_x2) and (v_y1 >= h_y2 - 80 or v_y2 >= h_y2):
                has_vest = True
                break
                
        if has_vest:
            compliant_count += 1
        else:
            infraction_count += 1

    # 6. Renderizado: Se convierte de BGR a RGB para corregir el tono de la piel en pantalla
    col1, col2 = st.columns([3, 1])
    
    with col1:
        res_plotted_bgr = filtered_res.plot()
        # Conversión que restaura los colores humanos normales para el usuario
        res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
        st.image(res_plotted_rgb, caption="Detección de Equipos de Protección Personal", use_container_width=True)
        
    with col2:
        st.subheader("📊 Auditoría de Cumplimiento SST")
        
        if infraction_count > 0:
            st.error(f"🚨 ALERTA DE SEGURIDAD\n\nSe detectaron **{infraction_count} trabajador(es)** sin chaleco de seguridad.")
        else:
            st.success("🟢 CUMPLIMIENTO CORRECTO\n\nTodos los trabajadores detectados portan chaleco y casco.")
            
        st.write("---")
        st.subheader("Conteo de EPP")
        
        counts = {'helmet': 0, 'vest': 0, 'boots': 0, 'gloves': 0, 'goggles': 0}
        for idx in valid_indices:
            c_name = res.names[int(boxes[idx].cls[0])]
            if c_name in counts:
                counts[c_name] += 1
            
        for name, count in counts.items():
            st.metric(label=name.upper(), value=count)
            
        st.write("---")
        df_report = pd.DataFrame([{
            'Cascos': counts['helmet'],
            'Chalecos': counts['vest'],
            'Botas': counts['boots'],
            'Guantes': counts['gloves'],
            'Gafas': counts['goggles'],
            'Conformes': compliant_count,
            'Infracciones': infraction_count
        }])
        
        csv_data = df_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Descargar Informe Auditoría CSV",
            data=csv_data,
            file_name="reporte_auditoria_sst.csv",
            mime="text/csv"
        )
