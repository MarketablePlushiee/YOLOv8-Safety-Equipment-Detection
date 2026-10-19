import streamlit as st
from ultralytics import YOLO
import PIL.Image

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Sistema de Monitoreo & Detección de EPP")

# Sidebar con sensibilidad por defecto en 0.20
conf_thresh = st.sidebar.slider("Sensibilidad (Umbral)", 0.10, 0.80, 0.20, 0.05)
iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.40, 0.05)

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    
    results = model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=800,      # Aumenta el detalle para prendas no reflectantes o lejanas
        augment=True    # Activa Test-Time Augmentation (TTA) para evaluar múltiples escalas
    )
    
    res_plotted = results[0].plot()
    
    # Corrección del parámetro para evitar el aviso en consola
    st.image(res_plotted, caption="Detección de EPP", use_container_width=True)
