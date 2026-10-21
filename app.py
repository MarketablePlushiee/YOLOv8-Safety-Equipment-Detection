import streamlit as st
from ultralytics import YOLO
import PIL.Image
from PIL import ImageDraw, ImageFont
import numpy as np
import pandas as pd

# Configuración de página
st.set_page_config(
    page_title="PPE Audit & Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    div[data-testid="stSidebarUserContent"] div.stRadio > label {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    
    div[data-testid="stSidebar"] div.stElementContainer:has(div.stRadio) {
        background-color: #1e293b;
        padding: 14px;
        border-radius: 10px;
        border: 2px solid #38bdf8;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"] div {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DICCIONARIO DE INTERNACIONALIZACIÓN (i18n)
# ---------------------------------------------------------
TEXTS = {
    'ES': {
        'page_title': "Sistema de Auditoría & Monitoreo de EPP",
        'subtitle': "Detección automatizada de Equipos de Protección Personal mediante Visión por Computadora (YOLOv8)",
        'expander_title': "ℹ️ Instrucciones de uso y guía de la aplicación",
        'guide_1': "1. **Cargar Imagen:** Sube una fotografía de inspección industrial (`.jpg`, `.jpeg`, `.png`).",
        'guide_2': "2. **Configuración Lateral:**",
        'guide_2a': "   * **Filtrado por Distancia:** Ignora automáticamente personas en el fondo lejano para evitar falsos positivos.",
        'guide_2b': "   * **Sensibilidad Adaptativa:** Ajusta dinámicamente los umbrales de confianza según la foto.",
        'guide_2c': "   * **Modo TTA (Multiescala):** Aplica aumento en inferencia para mejorar la detección de objetos complejos.",
        'guide_3': "3. **Resultados:** Revisa la imagen anotada, el estado de cumplimiento SST y descarga el informe en formato CSV.",
        'sidebar_lang': "🌐 LANGUAGE / IDIOMA",
        'sidebar_config': "⚙️ Configuración de Inferencia",
        'distant_label': "👥 Ignorar personas lejanas en Auditoría SST",
        'distant_help': "Excluye cascos pequeños en el fondo lejano para evitar falsas alarmas por distancia.",
        'auto_label': "🤖 Sensibilidad Automática Adaptativa (Por Foto)",
        'slider_gen': "Sensibilidad Cascos / Chalecos",
        'slider_small': "Sensibilidad Botas / Accesorios",
        'iou_label': "Solapamiento (IoU)",
        'tta_label': "Modo TTA (Multiescala)",
        'upload_label': "Subir fotografía de inspección industrial...",
        'audit_header': "📊 Auditoría de Cumplimiento SST",
        'alert_danger': "🚨 ALERTA DE SEGURIDAD\n\nSe detectaron **{count} trabajador(es)** sin chaleco de seguridad.",
        'alert_success': "🟢 CUMPLIMIENTO CORRECTO\n\nTodos los trabajadores evaluados portan chaleco y casco.",
        'distant_notice': "ℹ️ {count} sujeto(s) lejanos excluidos por distancia.",
        'count_header': "Conteo de EPP",
        'download_btn': "📄 Descargar Informe Auditoría CSV",
        'caption_img': "Detección de Equipos de Protección Personal",
        'classes': {
            'helmet': 'CASCOS',
            'vest': 'CHALECOS',
            'boots': 'BOTAS',
            'gloves': 'GUANTES',
            'goggles': 'GAFAS'
        },
        'csv_cols': ['Cascos', 'Chalecos', 'Botas', 'Guantes', 'Gafas', 'Conformes', 'Infracciones']
    },
    'EN': {
        'page_title': "PPE Audit & Monitoring System",
        'subtitle': "Automated Personal Protective Equipment Detection using Computer Vision (YOLOv8)",
        'expander_title': "ℹ️ Usage Instructions & Application Guide",
        'guide_1': "1. **Upload Image:** Upload an industrial inspection photograph (`.jpg`, `.jpeg`, `.png`).",
        'guide_2': "2. **Sidebar Settings:**",
        'guide_2a': "   * **Distance Filtering:** Automatically ignores background individuals to prevent false positives.",
        'guide_2b': "   * **Adaptive Sensitivity:** Dynamically adjusts confidence thresholds based on image detail.",
        'guide_2c': "   * **TTA Mode (Multi-scale):** Applies test-time augmentation to enhance detection performance.",
        'guide_3': "3. **Results:** Review annotated image, OHS compliance status, and download the CSV report.",
        'sidebar_lang': "🌐 LANGUAGE / IDIOMA",
        'sidebar_config': "⚙️ Inference Settings",
        'distant_label': "👥 Ignore distant workers in OHS Audit",
        'distant_help': "Excludes small helmets in the far background to prevent distance false alarms.",
        'auto_label': "🤖 Adaptive Auto Sensitivity (Per Photo)",
        'slider_gen': "Helmets / Vests Sensitivity",
        'slider_small': "Boots / Accessories Sensitivity",
        'iou_label': "Overlap (IoU)",
        'tta_label': "TTA Mode (Multi-scale)",
        'upload_label': "Upload industrial inspection photo...",
        'audit_header': "📊 OHS Compliance Audit",
        'alert_danger': "🚨 SAFETY ALERT\n\nDetected **{count} worker(s)** without safety vest.",
        'alert_success': "🟢 FULL COMPLIANCE\n\nAll evaluated workers are wearing safety vest and helmet.",
        'distant_notice': "ℹ️ {count} distant worker(s) excluded by distance threshold.",
        'count_header': "PPE Count",
        'download_btn': "📄 Download CSV Audit Report",
        'caption_img': "Personal Protective Equipment Detection",
        'classes': {
            'helmet': 'HELMETS',
            'vest': 'VESTS',
            'boots': 'BOOTS',
            'gloves': 'GLOVES',
            'goggles': 'GOGGLES'
        },
        'csv_cols': ['Helmets', 'Vests', 'Boots', 'Gloves', 'Goggles', 'Compliant', 'Infractions']
    }
}

# ---------------------------------------------------------
# SIDEBAR - SELECTOR DE IDIOMA
# ---------------------------------------------------------
lang_choice = st.sidebar.radio(
    "🌐 LANGUAGE / IDIOMA", 
    ["Español", "English"], 
    index=0,
    horizontal=True
)

lang_key = 'ES' if lang_choice == "Español" else 'EN'
t = TEXTS[lang_key]

st.title(f"🛡️ {t['page_title']}")
st.caption(t['subtitle'])

with st.expander(t['expander_title']):
    st.markdown(f"{t['guide_1']}\n{t['guide_2']}\n{t['guide_2a']}\n{t['guide_2b']}\n{t['guide_2c']}\n{t['guide_3']}")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

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
    img_pil = PIL.Image.fromarray(img_np).convert("RGBA")
    overlay = PIL.Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
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

    combined = PIL.Image.alpha_composite(img_pil, overlay).convert("RGB")
    return np.array(combined)

# Sidebar
st.sidebar.header(t['sidebar_config'])

ignore_distant = st.sidebar.checkbox(
    t['distant_label'], 
    value=True,
    help=t['distant_help']
)

auto_mode = st.sidebar.checkbox(t['auto_label'], value=True)

if not auto_mode:
    conf_general_user = st.sidebar.slider(t['slider_gen'], 0.05, 0.80, 0.15, 0.05)
    conf_small_user = st.sidebar.slider(t['slider_small'], 0.01, 0.50, 0.03, 0.01)

iou_thresh = st.sidebar.slider(t['iou_label'], 0.10, 0.80, 0.25, 0.05)
use_tta = st.sidebar.checkbox(t['tta_label'], value=True)

uploaded_file = st.file_uploader(t['upload_label'], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_height = img_np.shape[0]
    
    all_detections = []
    
    results = model.predict(source=image, conf=0.02, iou=iou_thresh, imgsz=1024, augment=use_tta, agnostic_nms=True)
    res = results[0]
    for box in res.boxes:
        all_detections.append({
            'xyxy': box.xyxy[0].cpu().numpy().tolist(),
            'class': res.names[int(box.cls[0])],
            'score': float(box.conf[0])
        })
            
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
        
        if c_name == 'boots':
            is_glove_conflict = False
            for g_xyxy in glove_boxes:
                if compute_iou(xyxy, g_xyxy) > 0.10:
                    is_glove_conflict = True
                    break
            if is_glove_conflict:
                continue
                
            if (xyxy[1] < img_height * 0.40) and score < 0.60:
                continue

        valid_detections.append(det)
        if c_name == 'helmet':
            helmet_boxes.append((xyxy, score))
        elif c_name == 'vest':
            vest_boxes.append((xyxy, score))

    compliant_count = 0
    infraction_count = 0
    ignored_distant = 0
    
    for h_xyxy, h_score in helmet_boxes:
        hx1, hy1, hx2, hy2 = h_xyxy
        h_height = hy2 - hy1
        h_x_center = (hx1 + hx2) / 2.0
        
        if ignore_distant and (h_height < img_height * 0.045):
            ignored_distant += 1
            continue
            
        has_vest = False
        for v_xyxy, v_score in vest_boxes:
            vx1, vy1, vx2, vy2 = v_xyxy
            v_x_center = (vx1 + vx2) / 2.0
            h_width = hx2 - hx1
            v_width = vx2 - vx1
            
            horizontal_near = (vx1 - h_width*0.6 <= h_x_center <= vx2 + h_width*0.6) or (hx1 - v_width*0.6 <= v_x_center <= hx2 + v_width*0.6)
            vertical_near = (vy1 >= hy1 - 10) and (vy1 <= hy2 + h_height * 5.0)
            
            if horizontal_near and vertical_near:
                has_vest = True
                break
                
        if has_vest:
            compliant_count += 1
        else:
            infraction_count += 1

    col1, col2 = st.columns([3, 1])
    
    with col1:
        rendered_image = draw_custom_boxes(img_np, valid_detections, CLASS_COLORS)
        st.image(rendered_image, caption=t['caption_img'], use_container_width=True)
        
    with col2:
        st.subheader(t['audit_header'])
        
        if infraction_count > 0:
            st.error(t['alert_danger'].format(count=infraction_count))
        else:
            st.success(t['alert_success'])
            
        if ignored_distant > 0:
            st.caption(t['distant_notice'].format(count=ignored_distant))
            
        st.write("---")
        st.subheader(t['count_header'])
        
        counts = {'helmet': 0, 'vest': 0, 'boots': 0, 'gloves': 0, 'goggles': 0}
        for det in valid_detections:
            c_name = det['class']
            if c_name in counts:
                counts[c_name] += 1
            
        for name, count in counts.items():
            label_translated = t['classes'].get(name, name.upper())
            st.metric(label=label_translated, value=count)
            
        st.write("---")
        cols_csv = t['csv_cols']
        df_report = pd.DataFrame([{
            cols_csv[0]: counts['helmet'],
            cols_csv[1]: counts['vest'],
            cols_csv[2]: counts['boots'],
            cols_csv[3]: counts['gloves'],
            cols_csv[4]: counts['goggles'],
            cols_csv[5]: compliant_count,
            cols_csv[6]: infraction_count
        }])
        
        csv_data = df_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=t['download_btn'],
            data=csv_data,
            file_name="ppe_audit_report.csv" if lang_key == 'EN' else "reporte_auditoria_sst.csv",
            mime="text/csv"
        )
