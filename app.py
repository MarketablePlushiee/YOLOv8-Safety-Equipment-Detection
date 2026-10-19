import streamlit as st
from ultralytics import YOLO
import PIL.Image

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Sistema de Monitoreo & Detección de EPP")

# Sidebar de controles
st.sidebar.header("Configuración de Inferencia")
conf_general = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.15, 0.80, 0.25, 0.05)
conf_small = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.05, 0.50, 0.08, 0.01)
iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.40, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Mayor precisión en sombras)", value=True)

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    
    # Inferencia a alta resolución con umbral base permisivo
    results = model.predict(
        source=image,
        conf=min(conf_general, conf_small),
        iou=iou_thresh,
        imgsz=1024,
        augment=use_tta
    )
    
    # Filtrado adaptativo de cajas por categoría
    res = results[0]
    boxes = res.boxes
    valid_indices = []
    
    # Nombres de las clases: 0: helmet, 1: vest, 2: boots, 3: gloves, 4: goggles
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        class_name = res.names[cls_id]
        
        # Aplicar el umbral correspondiente a la categoría
        if class_name in ['boots', 'gloves', 'goggles']:
            if score >= conf_small:
                valid_indices.append(idx)
        else:
            if score >= conf_general:
                valid_indices.append(idx)
    
    # Conservar únicamente las detecciones válidas
    filtered_res = res[valid_indices]
    
    # Generar y mostrar la imagen con el parámetro actualizado para evitar avisos
    res_plotted = filtered_res.plot()
    st.image(res_plotted, caption="Detección de EPP Optimizada", use_container_width=True)
