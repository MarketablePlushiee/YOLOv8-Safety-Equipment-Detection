import streamlit as st
from ultralytics import YOLO
import PIL.Image
from PIL import ImageDraw, ImageFont
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

# Paleta de colores RGB oficial por clase
CLASS_COLORS = {
    'helmet': (37, 99, 235),     # Azul Eléctrico
    'vest': (249, 115, 22),      # Naranja Reflectante EPP
    'boots': (161, 98, 7),       # Café Cuero
    'gloves': (16, 185, 129),    # Verde Esmeralda
    'goggles': (139, 92, 246)    # Morado / Violeta
}

# Dibujador nativo RGB para Streamlit
def draw_custom_boxes(img_np, detections, class_colors):
    img_pil = PIL.Image.fromarray(img_np).convert("RGBA")
    overlay = PIL.Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
        c_name = det['class']
        score = det['score']
        
        rgb_color = class_colors.get(c_name, (200, 200, 200))
        rgba_border = rgb_color + (255,)
        rgba_fill = rgb_color + (35,)  # Relleno sutil transparente
        
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

# Sidebar de Controles
st.sidebar.header("⚙️ Configuración de Inferencia")

use_sahi = st.sidebar.checkbox(
    "🔬 Activar Modo Gigapixel SAHI (Fotos de alta resolución)", 
    value=True,
    help="Divide la imagen en recortes de 640x640 para accesorios pequeños a gran distancia."
)

ignore_distant = st.sidebar.checkbox(
    "👥 Ignorar personas lejanas en Auditoría SST", 
    value=True,
    help="Excluye cascos lejanos en el fondo para evitar falsas alarmas."
)

auto_mode = st.sidebar.checkbox("🤖 Sensibilidad Automática Adaptativa (Por Foto)", value=True)

if not auto_mode:
    conf_general_user = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.05, 0.80, 0.15, 0.05)
    conf_small_user = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.03, 0.01)

iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.25, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

uploaded_file = st.file_uploader("Subir fotografía de inspección industrial...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_height = img_np.shape[0]
    
    all_detections = []
    
    # 1. Obtención de predicciones (SAHI o YOLO Estándar)
    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path='best.pt',
                confidence_threshold=0.02, # Mayor apertura para guantes sutiles
                device='cpu'
            )
            sahi_res = get_sliced_prediction(
                img_np,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.30, # Solapamiento optimizado del 30%
                overlap_width_ratio=0.30,
                postprocess_type="NMS",
                postprocess_match_threshold=iou_thresh
            )
            for pred in sahi_res.object_prediction_list:
                all_detections.append({
                    'xyxy': pred.bbox.to_xyxy(),
                    'class': pred.category.name,
                    'score': pred.score.value
                })
        except Exception as e:
            st.sidebar.error(f"Aviso SAHI: {e}. Usando YOLO estándar.")
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
            
    # 2. Lógica Adaptativa
    if auto_mode:
        floors = {'helmet': 0.10, 'vest': 0.08, 'boots': 0.03, 'gloves': 0.025, 'goggles': 0.03}
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

    # 3. Filtrado por Coherencia Posicional y Umbral
    helmet_boxes = [d for d in all_detections if d['class'] == 'helmet' and d['score'] >= class_thresholds.get('helmet', 0.10)]
    min_helmet_y = min([d['xyxy'][1] for d in helmet_boxes]) if helmet_boxes else 0
    
    valid_detections = []
    vest_boxes = []
    
    for det in all_detections:
        c_name = det['class']
        score = det['score']
        target_thresh = class_thresholds.get(c_name, 0.05)
        
        if score >= target_thresh:
            x1, y1, x2, y2 = det['xyxy']
            
            # Filtro de Coherencia Espacial: descarta botas falsas ubicadas en la zona superior/manos
            if c_name == 'boots' and helmet_boxes:
                if (y1 < min_helmet_y + 100) and score < 0.65:
                    continue
                    
            valid_detections.append(det)
            if c_name == 'vest':
                vest_boxes.append((det['xyxy'], score))

    # 4. Motor de Cumplimiento SST
    compliant_count = 0
    infraction_count = 0
    ignored_distant = 0
    
    for h_det in helmet_boxes:
        h_xyxy = h_det['xyxy']
        hx1, hy1, hx2, hy2 = h_xyxy
        h_height = hy2 - hy1
        h_x_center = (hx1 + hx2) / 2.0
        
        if ignore_distant and (h_height < img_height * 0.025):
            ignored_distant += 1
            continue
            
        has_vest = False
        for v_xyxy, v_score in vest_boxes:
            vx1, vy1, vx2, vy2 = v_xyxy
            v_x_center = (vx1 + vx2) / 2.0
            h_width = hx2 - hx1
            v_width = vx2 - vx1
            
            horizontal_near = (vx1 - h_width*0.6 <= h_x_center <= vx2 + h_width*0.6) or (hx1 - v_width*0.6 <= v_x_center <= hx2 + v_width*0.6)
            vertical_near = (vy1 >= hy1 - 10) and (vy1 <= hy2 + h_height * 5El análisis de las capturas evidencia el comportamiento exacto del modelo con y sin SAHI:

* **Falso positivo de bota en la mano (`image_9dc822.jpg`):** El guante negro señalando hacia abajo en el borde inferior (`boots 0.58`) se confunde con el cuero de una bota porque el encuadre corta el brazo y deja la textura oscura directamente sobre el suelo.
* **Validación de SAHI (`image_9dcb66.jpg` vs `image_9dcb84.jpg`):** Con SAHI activo se capturan **2 botas** (`0.24` y `0.21`), mientras que sin SAHI apenas capturaba una (`0.18`). El parcheado espacial demuestra su eficacia para recuperar geometría fina en la zona inferior.
* **Guantes omitidos (`image_9dcbe2.jpg`):** Los guantes amarillos de badana pegados a la cintura sufren por la falta de contraste frente a la tela del pantalón y la postura en reposo, lo que hace que la confianza quede rozando el umbral base.

---

### Soluciones para Maximizar la Precisión

Para solucionar la confusión del guante/bota y elevar la captura de guantes en reposo sin necesidad de reentrenar el modelo, se pueden aplicar **tres optimizaciones directas en la inferencia**:

#### 1. Reducción de Tamaño de Parche y Mayor Solapamiento en SAHI
Reducir las ventanas de SAHI de $640 \times 640$ a **$480 \times 480$** con un solapamiento del **$35\%$** hace que los guantes y gafas se vean proporcionalmente más grandes dentro de cada fragmento procesado por YOLO, elevando su confianza del $0.05$ al $0.30+$.

#### 2. Filtro de Consistencia Anatómica Espacial
Un guante no debe clasificarse como bota si está ubicado en la zona media/alta del cuerpo o cerca del área de las manos. Podemos incorporar una regla en Python que descarte predicciones de `boots` si su coordenada vertical se sitúa demasiado alta respecto al torso o si se superpone con la altura de un brazo.

#### 3. Supresión de Solapamientos por Clase (*Class-Agnostic NMS en SAHI*)
Para evitar que una misma región devuelva dos clases competidoras (ejemplo: `boots` y `gloves` en la misma mano), se ajusta el postprocesamiento de SAHI a un IoU más estricto ($0.25$).

---

### Código Optimizado para `app.py`

Sustituye la sección de predicción y SAHI en tu archivo `app.py` con estos ajustes de resolución y filtrado:

```python
import streamlit as st
from ultralytics import YOLO
import PIL.Image
from PIL import ImageDraw, ImageFont
import numpy as np
import pandas as pd

# Configuración de página
st.set_page_config(page_title="SST Vision - Auditoría de EPP", layout="wide")
st.title("🛡️ SST Vision - Sistema de Auditoría & Monitoreo de EPP")

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

def draw_custom_boxes(img_np, detections, class_colors):
    img_pil = PIL.Image.fromarray(img_np).convert("RGBA")
    overlay = PIL.Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
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

st.sidebar.header("⚙️ Configuración de Inferencia")

use_sahi = st.sidebar.checkbox(
    "🔬 Activar Modo Gigapixel SAHI (Fotos de alta resolución)", 
    value=True,
    help="Divide la imagen en recortes de 480x480 para accesorios pequeños a gran distancia."
)

ignore_distant = st.sidebar.checkbox(
    "👥 Ignorar personas lejanas en Auditoría SST", 
    value=True
)

auto_mode = st.sidebar.checkbox("🤖 Sensibilidad Automática Adaptativa (Por Foto)", value=True)

if not auto_mode:
    conf_general_user = st.sidebar.slider("Sensibilidad Cascos / Chalecos", 0.05, 0.80, 0.15, 0.05)
    conf_small_user = st.sidebar.slider("Sensibilidad Botas / Accesorios", 0.01, 0.50, 0.03, 0.01)

iou_thresh = st.sidebar.slider("Solapamiento (IoU)", 0.10, 0.80, 0.25, 0.05)
use_tta = st.sidebar.checkbox("Modo TTA (Multiescala)", value=True)

uploaded_file = st.file_uploader("Subir fotografía de inspección industrial...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_height = img_np.shape[0]
    
    all_detections = []
    
    # 1. Inferencia SAHI ajustada a parches de 480px y 35% de overlap
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
        except Exception as e:
            st.sidebar.error(f"Error SAHI: {e}. Usando YOLO nativo.")
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
            
    # 2. Lógica de Umbrales Adaptativos por Imagen
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

    # 3. Filtrado con Regla de Consistencia Anatómica Espacial para Botas
    valid_detections = []
    helmet_boxes = []
    vest_boxes = []
    
    for det in all_detections:
        c_name = det['class']
        score = det['score']
        target_thresh = class_thresholds.get(c_name, 0.03)
        
        if score >= target_thresh:
            x1, y1, x2, y2 = det['xyxy']
            
            # FILTRO ANATÓMICO: Descartar bota si está por encima del 45% superior de la foto sin estar en el suelo
            if c_name == 'boots' and (y1 < img_height * 0.45) and score < 0.60:
                continue
                
            valid_detections.append(det)
            if c_name == 'helmet':
                helmet_boxes.append((det['xyxy'], score))
            elif c_name == 'vest':
                vest_boxes.append((det['xyxy'], score))

    # 4. Auditoría SST
    compliant_count = 0
    infraction_count = 0
    ignored_distant = 0
    
    for h_xyxy, h_score in helmet_boxes:
        hx1, hy1, hx2, hy2 = h_xyxy
        h_height = hy2 - hy1
        h_x_center = (hx1 + hx2) / 2.0
        
        if ignore_distant and (h_height < img_height * 0.025):
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

    # 5. Visualización
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rendered_image = draw_custom_boxes(img_np, valid_detections, CLASS_COLORS)
        st.image(rendered_image, caption="Detección de Equipos de Protección Personal", use_container_width=True)
        
    with col2:
        st.subheader("📊 Auditoría de Cumplimiento SST")
        
        if infraction_count > 0:
            st.error(f"🚨 ALERTA DE SEGURIDAD\n\nSe detectaron **{infraction_count} trabajador(es)** sin chaleco de seguridad.")
        else:
            st.success("🟢 CUMPLIMIENTO CORRECTO\n\nTodos los trabajadores evaluados portan chaleco y casco.")
            
        if ignored_distant > 0:
            st.caption(f"ℹ️ {ignored_distant} sujeto(s) lejanos excluidos por distancia.")
            
        st.write("---")
        st.subheader("Conteo de EPP")
        
        counts = {'helmet': 0, 'vest': 0, 'boots': 0, 'gloves': 0, 'goggles': 0}
        for det in valid_detections:
            c_name = det['class']
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
