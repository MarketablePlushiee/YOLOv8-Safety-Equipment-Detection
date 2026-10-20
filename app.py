import streamlit as st
from ultralytics import YOLO
import PIL.Image

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.set_page_config(page_title="Detector de EPP - SST Vision", layout="wide")
st.title("Sistema de Auditoría & Monitoreo de EPP (SST)")

# Sidebar con parámetros óptimos preestablecidos
st.sidebar.header("Panel de Control de Inferencia")
conf_general = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.10, 0.80, 0.20, 0.05)
conf_small = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.05, 0.50, 0.10, 0.01)
iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.35, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala alta precisión)", value=True)

uploaded_file = st.file_uploader("Subir imagen de inspección...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    
    # Inferencia optimizada
    results = model.predict(
        source=image,
        conf=min(conf_general, conf_small),
        iou=iou_thresh,
        imgsz=1024,
        augment=use_tta,
        agnostic_nms=True
    )
    
    res = results[0]
    boxes = res.boxes
    valid_indices = []
    
    # Filtrado inteligente según tipo de objeto
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        class_name = res.names[cls_id]
        
        if class_name in ['boots', 'gloves', 'goggles']:
            if score >= conf_small:
                valid_indices.append(idx)
        else:
            if score >= conf_general:
                valid_indices.append(idx)
                
    filtered_res = res[valid_indices]
    
    # Mostrar resultados
    col1, col2 = st.columns([3, 1])
    
    with col1:
        res_plotted = filtered_res.plot()
        st.image(res_plotted, caption="Detección de Equipos de Protección Personal", use_container_width=True)
        
    with col2:
        st.subheader("Resumen de Conteo")
        counts = {}
        for idx in valid_indices:
            c_name = res.names[int(boxes[idx].cls[0])]
            counts[c_name] = counts.get(c_name, 0) + 1
            
        for name, count in counts.items():
            st.metric(label=name.upper(), value=count)
