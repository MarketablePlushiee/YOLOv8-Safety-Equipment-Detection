import streamlit as st
from ultralytics import YOLO
import PIL.Image
import cv2
import numpy as np

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Sistema de Monitoreo & Detección de EPP")

# Sidebar de controles
st.sidebar.header("Configuración de Inferencia")
conf_general = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.10, 0.80, 0.15, 0.05)
conf_small = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.05, 0.01)
iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.35, 0.05)

enhance_shadows = st.sidebar.checkbox("Realce Automático de Sombras (CLAHE)", value=True)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    # Realce de contraste en zonas oscuras para calzado/accesorios
    if enhance_shadows:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        image = PIL.Image.fromarray(img_np)

    # Inferencia optimizada con NMS agnóstico para evitar cajas encimadas
    results = model.predict(
        source=image,
        conf=min(conf_general, conf_small),
        iou=iou_thresh,
        imgsz=1024,
        augment=use_tta,
        agnostic_nms=True
    )
    
    # Filtrado adaptativo
    res = results[0]
    boxes = res.boxes
    valid_indices = []
    
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
    
    # Renderizado final
    res_plotted = filtered_res.plot()
    st.image(res_plotted, caption="Detección de EPP Optimizada", use_container_width=True)
