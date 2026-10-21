# 🛡️ PPE Audit & Monitoring System (YOLOv8)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red?style=for-the-badge&logo=streamlit)](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Medium-orange?style=for-the-badge&logo=ultralytics)](https://ultralytics.com/)

Aplicación web para la detección automatizada de Equipos de Protección Personal (EPP) y auditoría de cumplimiento en Seguridad y Salud en el Trabajo (SST). Utiliza el modelo **YOLOv8 Medium** ajustado para visión por computadora en entornos industriales.

---

## 📋 1. Documentación del Dataset

El modelo fue entrenado utilizando un dataset de **1,518 imágenes de inspección industrial** recopiladas y consolidadas desde repositorios especializados en seguridad laboral en Roboflow Universe.

* **Total de imágenes:** 1,518 fotografías anotadas.
* **División del Dataset:**
  * **Entrenamiento (Train):** 70% (~1,062 imágenes).
  * **Validación (Val):** 20% (~304 imágenes).
  * **Prueba (Test):** 10% (~152 imágenes).
* **Clases objetivo (5 clases):**
  1. `helmet` (Casco de seguridad)
  2. `vest` (Chaleco reflectante)
  3. `boots` (Calzado de seguridad)
  4. `gloves` (Guantes de protección)
  5. `goggles` (Gafas / Antiparras de protección)

---

## ⚙️ 2. Arquitectura y Lógica de Cumplimiento SST

El sistema procesa las imágenes mediante la siguiente canalización:

1. **Inferencia de Detección:** Ejecución directa unipaso sobre la imagen reescalada a 1024px con el modelo `YOLOv8m`.
2. **Filtrado por Distancia:** Se descartan automáticamente cascos con un alto relativo menor al 4.5% de la altura total de la imagen para evitar falsos positivos con personas en segundo plano lejano.
3. **Asociación Espacial por Trabajador:** Se aplica una regla de intersección delimitadora entre cascos e indumentaria. Un sujeto es evaluado como **Conforme** si existe un chaleco (`vest`) cuya coordenada horizontal colisione con la proyección vertical del casco (`helmet`) a una distancia no mayor a 5 veces la altura del casco.
4. **Sensibilidad Adaptativa:** Ajuste dinámico de umbrales por imagen según la distribución de puntajes de confianza máximos para equilibrar el recall en accesorios pequeños (`boots`, `gloves`, `goggles`).

### ⚠️ Limitaciones Técnicas y Casos de Falla
* **Oclusión parcial o severa:** Pérdida de detección cuando el chaleco o casco se encuentra cubierto por maquinaria u objetos en más de un 60%.
* **Baja iluminación:** Reducción en la tasa de detección (recall) en tomas nocturnas o espacios subterráneos sin luz directa.
* **Objetos sub-15px:** Detección limitada en elementos muy pequeños (como gafas o guantes) cuando la resolución efectiva del objeto cae por debajo de los 15 píxeles en la toma.

---

## 📊 3. Métricas Cuantitativas de Rendimiento

Resultados obtenidos en la evaluación del conjunto de prueba (152 imágenes en `results.csv` tras 50 épocas de entrenamiento):

| Métrica General | Valor Cuantitativo |
| :--- | :--- |
| **mAP@50** | **75.1%** (0.751) |
| **mAP@50-95** | **48.0%** (0.480) |
| **Recall (Exhaustividad)** | **75.6%** (0.756) |
| **Precision (Precisión)** | **68.1%** (0.681) |

### Rendimiento Detallado por Clase (mAP@50)
* **Cascos (`helmet`):** 88.0% mAP@50
* **Chalecos (`vest`):** 84.0% mAP@50
* **Botas (`boots`):** 72.0% mAP@50
* **Guantes (`gloves`):** 68.0% mAP@50
* **Gafas (`goggles`):** 64.0% mAP@50

### 📈 Gráficos de Evaluación
* **Convergencia de Pérdidas (`results.png`):** Disminución sostenida de la pérdida de caja (*box loss*) de 1.62 a 0.93 en 50 épocas sin evidencia de sobreajuste.
* **Curva Precision-Recall (`BoxPR_curve.png`):** Desempeño destacado en las clases de mayor volumen espacial (cascos y chalecos).
* **Matriz de Confusión (`confusion_matrix.png`):** Alta concentración de verdaderos positivos en la diagonal principal; confusión residual ubicada en guantes pequeños frente al fondo.

---

## 🔄 4. Justificación Técnica del Modelo e Inferencia

* **Selección de Modelo (YOLOv8m vs YOLOv8s):** Se comparó el desempeño de `YOLOv8s` (Small) frente a `YOLOv8m` (Medium). Aunque `YOLOv8s` reduce la latencia en un 25%, `YOLOv8m` obtuvo un incremento de **+6.2% en mAP@50** en clases pequeñas (`gloves` y `goggles`), justificando su elección para auditorías de precisión en entornos industriales.
* **Decisión sobre Inferencia Unipaso vs SAHI:** Se evaluó la inferencia por fragmentación (SAHI). Aunque la división en teselas incrementa la detección en bordes lejanos, genera un incremento de tiempo de procesamiento de más de **4x en servidor CPU** y posibles falsos positivos por pérdida de contexto global. Se optó por inferencia directa unipaso a 1024px para mantener respuestas inmediatas en la nube.

---

## 📦 5. Instalación y Uso Local

1. Clonar el repositorio:
   git clone https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git
   cd yolov8-safety-equipment-detection

2. Instalar dependencias requeridas:
   pip install -r requirements.txt

3. Ejecutar la aplicación web en local:
   streamlit run app.py

---

## 👨‍💻 Autor y Contacto

* **Ivan Latrille Brito** — Estudiante de Ingeniería en Realidad Virtual y Diseño de Juegos Digitales, Universidad Bernardo O'Higgins.
* **Aplicación Desplegada:** [Streamlit Cloud Demo](https://yolov8-safety-equipment-detection-qsj9ufsc4gezwsv7v3t9ms.streamlit.app/)
* **Portafolio Web:** [marketableplushiee.github.io/Portafolio](https://marketableplushiee.github.io/Portafolio/)
