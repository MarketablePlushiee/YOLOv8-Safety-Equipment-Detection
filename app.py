import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import gc

# Configuración de página
st.set_page_config(page_title="SST Vision - Auditoría de EPP", layout="wide")
st.title("🛡️ SST Vision - Sistema de Auditoría & Monitoreo de EPP")

@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error al cargar el archivo de pesos 'best.pt': {e}")
        return None

model = load_model()

# Paleta de colores RGB oficial por clase
CLASS_COLORS = {
    'helmet': (37, 99, 235),     # Azul Eléctrico
    'vest': (249, 115, 22),      # Naranja Reflectante EPP
    'boots': (161, 98, 7),       # Café Cuero
    'gloves': (16, 185, 129),    # Verde Esmeralda
    'goggles': (139, 92, 246)    # Morado / Violeta
}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def draw_custom_boxes(img_np, detections, class_colors):
    img_pil = Image.fromarray(img_np).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
        c_name = det['class']
        score = det['score']
        
        rgb_color = class_colors.get(c_name, (200, 200, 200))
        rgba_border = rgb_color + (255,)
        rgba_fill = rgb_color + (35,)
        
        draw.rectangle([x1, y1, x2, y2], fill=rgba_fill, outline=rgba_border, width=3)
        
        label_text = f"{c_name} {score:.2f}"
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(label_text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = 80, 14
            
        banner_y1 = max(0, y1 - th - 8)
        banner_y2 = y1
        draw.rectangle([x1, banner_y1, x1 + tw + 10, banner_y2], fill=rgba_border)
        draw.text((x1 + 5, banner_y1 + 2), label_text, fill=(255, 255, 255, 255), font=font)

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return np.array(combined)

try:
    # Sidebar
    st.sidebar.header("⚙️ Configuración de Inferencia")

    use_sahi = st.sidebar.checkbox(
        "🔬 Activar Modo Gigapixel SAHI (Fotos de alta resolución)", 
        value=False,
        help="Divide la imagen en recortes de 480x480 para accesorios pequeños a gran distancia."
    )

    ignore_distant = st.sidebar.checkbox(
        "👥 Ignorar personas lejanas en Auditoría SST", 
        value=True,
        help="Excluye cascos pequeños en el fondo lejano para evitar falsas alarmas por distancia."
    )

    auto_mode = st.sidebar.checkbox("🤖 Sensibilidad Automática Adaptativa (Por Foto)", value=True)

    if not auto_mode:
        conf_general_user = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.05, 0.80, 0.15, 0.05)
        conf_small_user = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.03, 0.01)
    else:
        conf_general_user = 0.15
        conf_small_user = 0.03

    iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.25, 0.05)
    use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

    uploaded_file = st.file_uploader("Subir fotografía de inspección industrial...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_height = img_np.shape[0]
        
        all_detections = []
        
        # 1. Inferencia
        if use_sahi:
            try:
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction
                
                detection_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path='best.pt',
                    confidence_threshold=0.02,
                    device='cpu'
                )
                sahi_res = get_sliced_prediction(
                    img_np,
                    detection_model,
                    slice_height=480,
                    slice_width=480,
                    overlap_height_ratio=0.35,
                    overlap_width_ratio=0.35,
                    postprocess_type="NMS",
                    postprocess_match_threshold=iou_thresh
                )
                for pred in sahi_res.object_prediction_list:
                    all_detections.append({
                        'xyxy': pred.bbox.to_xyxy(),
                        'class': pred.category.name,
                        'score': pred.score.value
                    })
                del detection_model, sahi_res
                gc.collect()
            except Exception as e:
                st.sidebar.error(f"Aviso SAHI no disponible: {e}. Usando YOLO estándar.")
                results = model.predict(source=image, conf=0.02, iou=iou_thresh, imgsz=1024, augment=use_tta, agnostic_nms=True)
                res = results[0]
                for box in res.boxes:
                    all_detections.append({
                        'xyxy': box.xyxy[0].cpu().numpy().tolist(),
                        'class': res.names[int(box.cls[0])],
                        'score': float(box.conf[0])
                    })
        else:
            results = model.predict(source=image, conf=0.02, iou=iou_thresh, imgsz=1024, augment=use_tta, agnostic_nms=True)
            res = results[0]
            for box in res.boxes:
                all_detections.append({
                    'xyxy': box.xyxy[0].cpu().numpy().tolist(),
                    'class': res.names[int(box.cls[0])],
                    'score': float(box.conf[0])
                })
                
        # 2. Umbrales Adaptativos
        if auto_mode:
            floors = {'helmet': 0.10, 'vest': 0.08, 'boots': 0.03, 'gloves': 0.02, 'goggles': 0.03}
            max_scores = {k: 0.0 for k in floors.keys()}
            for det in all_detections:
                c_name = det['class']
                if c_name in max_scores and det['score'] > max_scores[c_name]:
                    max_scores[c_name] = det['score']
                    
            class_thresholds = {}
            for c_name, floor_val in floors.items():
                if max_scores[c_name] > 0:
                    class_thresholds[c_name] = max(floor_val, min(max_scores[c_name] * 0.30, 0.18))
                else:
                    class_thresholds[c_name] = floor_val
        else:
            class_thresholds = {
                'helmet': conf_general_user, 'vest': conf_general_user,
                'boots': conf_small_user, 'gloves': conf_small_user, 'goggles': conf_small_user
            }

        # 3. Filtrado con Regla de Exclusión Mutua (Mano vs Bota)
        candidate_detections = []
        glove_boxes = []
        
        for det in all_detections:
            c_name = det['class']
            score = det['score']
            target_thresh = class_thresholds.get(c_name, 0.03)
            if score >= target_thresh:
                candidate_detections.append(det)
                if c_name == 'gloves':
                    glove_boxes.append(det['xyxy'])

        valid_detections = []
        helmet_boxes = []
        vest_boxes = []

        for det in candidate_detections:
            c_name = det['class']
            score = det['score']
            xyxy = det['xyxy']
            
            # Elimina falsos positivos de botas sobre las manos
            if c_name == 'boots':
                is_glove_conflict = False
                for g_xyxy in glove_boxes:
                    if compute_iou(xyxy, g_xyxy) > 0.10:
                        is_glove_conflict = True
                        break
                if is_glove_conflict:
                    continue
                    
                # Filtro espacial por altura
                if (xyxy[1] < img_height * 0.40) and score < 0.60:
                    continue

            valid_detections.append(det)
            if c_name == 'helmet':
                helmet_boxes.append((xyxy, score))
            elif c_name == 'vest':
                vest_boxes.append((xyxy, score))

        # 4. Auditoría de Cumplimiento SST (Umbral lejanía al 4.5%)
        compliant_count = 0
        infraction_count = 0
        ignored_distant = 0
        
        for h_xyxy, h_score in helmet_boxes:
            hx1, hy1
