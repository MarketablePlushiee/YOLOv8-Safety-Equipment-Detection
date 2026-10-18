import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detector EPP - YOLOv8", layout="centered")

st.title("🛡️ Detector de Elementos de Protección Personal (EPP)")
st.write("Sube una imagen industrial para verificar la detección de EPP en tiempo real.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Deslizador para ajustar la sensibilidad del detector
conf_threshold = st.slider("Sensibilidad de detección (Umbral de confianza)", min_value=0.10, max_value=0.90, value=0.20, step=0.05)

uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen de entrada", use_container_width=True)
    
    with st.spinner("Procesando detección con YOLOv8..."):
        # Se pasa el umbral de confianza seleccionado
        results = model(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        res_rgb = res_plotted[:, :, ::-1]
        st.image(res_rgb, caption="Detección en tiempo real", use_container_width=True)
