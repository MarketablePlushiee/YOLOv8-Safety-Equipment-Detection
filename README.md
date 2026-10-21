# 🛡️ PPE Audit & Monitoring System (YOLOv8)

🌐 **Language / Idioma:** [English](#-ppe-audit--monitoring-system-yolov8---english) | [Español](#-sistema-de-auditoría-y-monitoreo-de-epp-yolov8---español)

---

## 🇬🇧 PPE Audit & Monitoring System (YOLOv8) - English

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red?style=for-the-badge&logo=streamlit)](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Medium-orange?style=for-the-badge&logo=ultralytics)](https://ultralytics.com/)

Web application for automated Personal Protective Equipment (PPE) detection and Occupational Safety and Health (OSH) compliance auditing. Built using a fine-tuned **YOLOv8 Medium** computer vision model optimized for industrial environments.

---

### 📋 1. Dataset Documentation & Consolidation

The model was trained on a large-scale master dataset named **`ppe-master`**, containing **29,087 annotated images**. This dataset was built by aggregating and standardizing 8 domain-specific occupational safety sub-projects hosted on Roboflow Universe.

* **Master Dataset Name:** `ppe-master` (Version 1)
* **Total Images:** 29,087 annotated photographs
* **Dataset Split:**
  * **Train Set:** 72% (20,918 images)
  * **Validation Set (Val):** 15% (4,347 images)
  * **Test Set:** 13% (3,822 images)

#### Source Projects & Sub-Datasets Aggregated
1. `Hard-Hats` — 7,035 images (Head protection)
2. `PPE detection` — 5,140 images (Multi-class industrial inspection)
3. `safety-boots` — 4,896 images (Footwear protection)
4. `vest` — 3,764 images (High-visibility clothing)
5. `Safety Gloves` — 3,373 images (Hand protection)
6. `VEST` — 2,127 images (Reflective safety gear)
7. `PPE DETECTION` — 2,114 images (Site compliance monitoring)
8. `glasses-ppe` — 638 images (Eye/Goggles protection)

* **Target Classes (5 primary OSH classes):**
  1. `helmet` (Safety Helmet / Hardhat)
  2. `vest` (High-Visibility Reflective Vest)
  3. `boots` (Safety Footwear)
  4. `gloves` (Protective Gloves)
  5. `goggles` (Safety Goggles / Glasses)

![Dataset Label Distribution](labels.jpg)

---

### ⚙️ 2. Architecture & OSH Compliance Logic

The system processes input images through the following pipeline:

1. **Detection Inference:** Single-pass direct inference on images resized to 1024px using `YOLOv8m`.
2. **Distance Filtering:** Hardhats with a relative height smaller than 4.5% of the total image height are automatically filtered out to eliminate distant background false positives.
3. **Worker Spatial Association:** A bounding-box intersection heuristic evaluates compliance per worker. An individual is classified as **Compliant** if a high-visibility vest (`vest`) overlaps horizontally with the vertical projection of a helmet (`helmet`) within a maximum distance of 5 times the helmet height.
4. **Adaptive Sensitivity:** Dynamic per-image confidence thresholding based on the top score distribution to maintain balanced recall across small equipment classes (`boots`, `gloves`, `goggles`).

#### ⚠️ Technical Limitations & Failure Modes
* **Partial or Severe Occlusion:** Detection accuracy degrades when vests or helmets are obscured by machinery or structural elements by more than 60%.
* **Low Illumination:** Reduced recall in nighttime shots or dimly lit underground environments without direct lighting.
* **Sub-15px Objects:** Limited detection rate on small items (such as gloves or goggles) when their effective bounding box dimension falls below 15 pixels.

---

### 📊 3. Quantitative Performance Metrics

Results evaluated on the test dataset (`results.csv` after 50 training epochs):

| Metric | Quantitative Value |
| :--- | :--- |
| **mAP@50** | **75.1%** (0.751) |
| **mAP@50-95** | **48.0%** (0.480) |
| **Recall** | **75.6%** (0.756) |
| **Precision** | **68.1%** (0.681) |

#### Per-Class Performance Breakdown (mAP@50)
* **Helmets (`helmet`):** 88.0% mAP@50
* **Vests (`vest`):** 84.0% mAP@50
* **Boots (`boots`):** 72.0% mAP@50
* **Gloves (`gloves`):** 68.0% mAP@50
* **Goggles (`goggles`):** 64.0% mAP@50

#### 📈 Training & Evaluation Plots

![Loss Convergence](results.png)
* **Loss Convergence (`results.png`):** Steady decrease in bounding box loss (*box loss*) from 1.62 down to 0.93 across 50 epochs without signs of overfitting.

![Precision-Recall Curve](BoxPR_curve.png)
* **Precision-Recall Curve (`BoxPR_curve.png`):** High precision sustained across primary spatial classes (helmets and vests).

![Confusion Matrix](confusion_matrix.png)
* **Confusion Matrix (`confusion_matrix.png`):** Strong diagonal concentration of true positives; minor background false positives confined to small gloves.

---

### 🔄 4. Technical Model & Inference Justification

* **Model Selection (YOLOv8m vs YOLOv8s):** Benchmark comparisons between `YOLOv8s` (Small) and `YOLOv8m` (Medium) showed that while `YOLOv8s` reduced latency by 25%, `YOLOv8m` yielded a **+6.2% mAP@50 gain** on smaller target classes (`gloves` and `goggles`), justifying its selection for high-accuracy safety audits.
* **Single-Pass Inference vs SAHI:** Sliced Aided Hyper Inference (SAHI) was evaluated. While tile slicing slightly improved edge object recall, it incurred a **>4x latency penalty on CPU instances** and increased false positive risk due to lost global contextual cues. Single-pass inference at 1024px was selected to maintain real-time responsiveness on web deployments.

---

### 📦 5. Local Installation & Setup

```bash
# 1. Clone repository
git clone [https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git](https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git)
cd yolov8-safety-equipment-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch local Streamlit interface
streamlit run app.py
```

---

### 👨‍💻 Author & Contact

* **Ivan Latrille Brito** — Virtual Reality Engineering and Digital Game Design Undergraduate Student, Universidad Bernardo O'Higgins.
* **Live Web App:** [Streamlit Cloud Demo](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
* **Portfolio:** [marketableplushiee.github.io/Portafolio](https://marketableplushiee.github.io/Portafolio/)

---
---

## 🇪🇸 Sistema de Auditoría y Monitoreo de EPP (YOLOv8) - Español

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red?style=for-the-badge&logo=streamlit)](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Medium-orange?style=for-the-badge&logo=ultralytics)](https://ultralytics.com/)

Aplicación web para la detección automatizada de Equipos de Protección Personal (EPP) y auditoría de cumplimiento en Seguridad y Salud en el Trabajo (SST). Utiliza el modelo **YOLOv8 Medium** ajustado para visión por computadora en entornos industriales.

---

### 📋 1. Documentación del Dataset y Consolidación

El modelo fue entrenado utilizando un dataset masivo denominado **`ppe-master`**, que reúne **29,087 imágenes anotadas**. Este conjunto fue construido consolidando y estandarizando 8 sub-proyectos especializados de seguridad laboral alojados en Roboflow Universe.

* **Nombre del Dataset Maestro:** `ppe-master` (Versión 1)
* **Total de imágenes:** 29,087 fotografías anotadas
* **División del Dataset:**
  * **Entrenamiento (Train):** 72% (20,918 imágenes)
  * **Validación (Val):** 15% (4,347 imágenes)
  * **Prueba (Test):** 13% (3,822 imágenes)

#### Sub-Proyectos Integrados
1. `Hard-Hats` — 7,035 imágenes (Protección de cabeza)
2. `PPE detection` — 5,140 imágenes (Inspección industrial multiclase)
3. `safety-boots` — 4,896 imágenes (Calzado de seguridad)
4. `vest` — 3,764 imágenes (Vestimenta de alta visibilidad)
5. `Safety Gloves` — 3,373 imágenes (Protección de manos)
6. `VEST` — 2,127 imágenes (Indumentaria reflectante)
7. `PPE DETECTION` — 2,114 imágenes (Monitoreo de cumplimiento en obra)
8. `glasses-ppe` — 638 imágenes (Protección ocular / Antiparras)

* **Clases objetivo (5 clases SST principales):**
  1. `helmet` (Casco de seguridad)
  2. `vest` (Chaleco reflectante)
  3. `boots` (Calzado de seguridad)
  4. `gloves` (Guantes de protección)
  5. `goggles` (Gafas / Antiparras de protección)

![Distribución de Clases](labels.jpg)

---

### ⚙️ 2. Arquitectura y Lógica de Cumplimiento SST

El sistema procesa las imágenes mediante la siguiente canalización:

1. **Inferencia de Detección:** Ejecución directa unipaso sobre la imagen reescalada a 1024px con el modelo `YOLOv8m`.
2. **Filtrado por Distancia:** Se descartan automáticamente cascos con un alto relativo menor al 4.5% de la altura total de la imagen para evitar falsos positivos con personas en segundo plano lejano.
3. **Asociación Espacial por Trabajador:** Se aplica una regla de intersección delimitadora entre cascos e indumentaria. Un sujeto es evaluado como **Conforme** si existe un chaleco (`vest`) cuya coordenada horizontal colisione con la proyección vertical del casco (`helmet`) a una distancia no mayor a 5 veces la altura del casco.
4. **Sensibilidad Adaptativa:** Ajuste dinámico de umbrales por imagen según la distribución de puntajes de confianza máximos para equilibrar el recall en accesorios pequeños (`boots`, `gloves`, `goggles`).

#### ⚠️ Limitaciones Técnicas y Casos de Falla
* **Oclusión parcial o severa:** Pérdida de detección cuando el chaleco o casco se encuentra cubierto por maquinaria u objetos en más de un 60%.
* **Baja iluminación:** Reducción en la tasa de detección (recall) en tomas nocturnas o espacios subterráneos sin luz directa.
* **Objetos sub-15px:** Detección limitada en elementos muy pequeños (como gafas o guantes) cuando la resolución efectiva del objeto cae por debajo de los 15 píxeles en la toma.

---

### 📊 3. Métricas Cuantitativas de Rendimiento

Resultados obtenidos en la evaluación del conjunto de prueba (`results.csv` tras 50 épocas de entrenamiento):

| Métrica General | Valor Cuantitativo |
| :--- | :--- |
| **mAP@50** | **75.1%** (0.751) |
| **mAP@50-95** | **48.0%** (0.480) |
| **Recall (Exhaustividad)** | **75.6%** (0.756) |
| **Precision (Precisión)** | **68.1%** (0.681) |

#### Rendimiento Detallado por Clase (mAP@50)
* **Cascos (`helmet`):** 88.0% mAP@50
* **Chalecos (`vest`):** 84.0% mAP@50
* **Botas (`boots`):** 72.0% mAP@50
* **Guantes (`gloves`):** 68.0% mAP@50
* **Gafas (`goggles`):** 64.0% mAP@50

#### 📈 Gráficos de Evaluación

![Convergencia de Pérdidas](results.png)
* **Convergencia de Pérdidas (`results.png`):** Disminución sustained de la pérdida de caja (*box loss*) de 1.62 a 0.93 en 50 épocas sin evidencia de sobreajuste.

![Curva Precision-Recall](BoxPR_curve.png)
* **Curva Precision-Recall (`BoxPR_curve.png`):** Desempeño destacado en las clases de mayor volumen espacial (cascos y chalecos).

![Matriz de Confusión](confusion_matrix.png)
* **Matriz de Confusión (`confusion_matrix.png`):** Alta concentración de verdaderos positivos en la diagonal principal; confusión residual ubicada en guantes pequeños frente al fondo.

---

### 🔄 4. Justificación Técnica del Modelo e Inferencia

* **Selección de Modelo (YOLOv8m vs YOLOv8s):** Se comparó el desempeño de `YOLOv8s` (Small) frente a `YOLOv8m` (Medium). Aunque `YOLOv8s` reduce la latencia en un 25%, `YOLOv8m` obtuvo un incremento de **+6.2% en mAP@50** en clases pequeñas (`gloves` y `goggles`), justificando su elección para auditorías de precisión en entornos industriales.
* **Decisión sobre Inferencia Unipaso vs SAHI:** Se evaluó la inferencia por fragmentación (SAHI). Aunque la división en teselas incrementa la detección en bordes lejanos, genera un incremento de tiempo de procesamiento de más de **4x en servidor CPU** y posibles falsos positivos por pérdida de contexto global. Se optó por inferencia directa unipaso a 1024px para mantener respuestas inmediatas en la nube.

---

### 📦 5. Instalación y Uso Local

```bash
# 1. Clonar el repositorio
git clone [https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git](https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git)
cd yolov8-safety-equipment-detection

# 2. Instalar dependencias requeridas
pip install -r requirements.txt

# 3. Ejecutar la aplicación web en local
streamlit run app.py
```

---

### 👨‍💻 Autor y Contacto

* **Ivan Latrille Brito** — Estudiante de Ingeniería en Realidad Virtual y Diseño de Juegos Digitales, Universidad Bernardo O'Higgins.
* **Aplicación Desplegada:** [Streamlit Cloud Demo](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
* **Portafolio Web:** [marketableplushiee.github.io/Portafolio](https://marketableplushiee.github.io/Portafolio/)
