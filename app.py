import streamlit as st
from ultralytics import YOLO
import PIL.Image
import pandas as pd

# Configuración inicial
st.set_page_config(page_title="SST Vision - Auditoría de EPP", layout="wide")
st.title("🛡️ SST Vision - Sistema de Auditoría & Monitoreo de EPP")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Panel de Control
st.sidebar.header("⚙️ Configuración de Inferencia")

# Modo Automático Adaptativo
auto_mode = st.sidebar.checkbox("🤖 Sensibilidad Automática Adaptativa (Por Foto)", value=True)

if auto_mode:
    st.sidebar.info("La sensibilidad se ajustará dinámicamente según la iluminación, distancia y contraste de cada imagen.")
    conf_general_user = 0.20
    conf_small_user = 0.05
else:
    conf_general_user = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.10, 0.80, 0.25, 0.05)
    conf_small_user = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.08, 0.01)

iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.30, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

uploaded_file = st.file_uploader("Subir fotografía de inspección industrial...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    
    # Inferencia inicial con umbral base permisivo
    results = model.predict(
        source=image,
        conf=0.03,  # Captura candidatos sutiles como guantes o gafas
        iou=iou_thresh,
        imgsz=1024,
        augment=use_tta,
        agnostic_nms=True
    )
    
    res = results[0]
    boxes = res.boxes
    
    # Lógica de Umbrales Adaptativos por Imagen
    if auto_mode:
        # Pisos mínimos base para evitar ruido extremo
        floors = {
            'helmet': 0.15,
            'vest': 0.15,
            'boots': 0.05,
            'gloves': 0.04,   # Permite capturar guantes reales de 0.07
            'goggles': 0.04
        }
        
        # Encontrar la confianza máxima registrada en esta foto por clase
        max_scores = {k: 0.0 for k in floors.keys()}
        for box in boxes:
            c_name = res.names[int(box.cls[0])]
            score = float(box.conf[0])
            if c_name in max_scores and score > max_scores[c_name]:
                max_scores[c_name] = score
                
        # Calcular el umbral dinámico para cada clase
        class_thresholds = {}
        for c_name, floor_val in floors.items():
            if max_scores[c_name] > 0:
                # El umbral sube si hay detecciones muy claras, pero no sobrepasa 0.25
                class_thresholds[c_name] = max(floor_val, min(max_scores[c_name] * 0.35, 0.25))
            else:
                class_thresholds[c_name] = floor_val
    else:
        # Valores manuales seleccionados en los sliders
        class_thresholds = {
            'helmet': conf_general_user,
            'vest': conf_general_user,
            'boots': conf_small_user,
            'gloves': conf_small_user,
            'goggles': conf_small_user
        }

    # Filtrar cajas según el umbral correspondiente
    valid_indices = []
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        class_name = res.names[cls_id]
        
        target_thresh = class_thresholds.get(class_name, 0.10)
        if score >= target_thresh:
            valid_indices.append(idx)
            
    filtered_res = res[valid_indices]
    
    # Renderizado y Visualización
    col1, col2 = st.columns([3, 1])
    
    with col1:
        res_plotted = filtered_res.plot()
        st.image(res_plotted, caption="Detección de Equipos de Protección Personal", use_container_width=True)
        
    with col2:
        st.subheader("📊 Resumen de Inspección")
        counts = {'helmet': 0, 'vest': 0, 'boots': 0, 'gloves': 0, 'goggles': 0}
        
        for idx in valid_indices:
            c_name = res.names[int(boxes[idx].cls[0])]
            if c_name in counts:
                counts[c_name] += 1
            
        for name, count in counts.items():
            st.metric(label=name.upper(), value=count)
            
        st.write("---")
        df_report = pd.DataFrame([counts])
        csv_data = df_report.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📄 Descargar Reporte CSV",
            data=csv_data,
            file_name="reporte_auditoria_epp.csv",
            mime="text/csv"
        )
