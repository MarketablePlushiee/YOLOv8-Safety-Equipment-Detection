import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detector EPP - YOLOv8", layout="centered")

st.title("🛡️ Detector de Elementos de Protección Personal (EPP)")
st.write("Sube una imagen industrial para detectar cascos, chalecos y calzado de seguridad en tiempo real.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen de entrada", use_container_width=True)
    
    with st.spinner("Procesando detección con YOLOv8..."):
        results = model(image)
        res_plotted = results[0].plot()
        # Convertir formato de color BGR a RGB
        res_rgb = res_plotted[:, :, ::-1]
        st.image(res_rgb, caption="Detección en tiempo real", use_container_width=True)
