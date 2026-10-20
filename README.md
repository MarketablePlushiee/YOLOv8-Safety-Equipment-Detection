# 🛡️ PPE Audit & Monitoring System (YOLOv8 + SAHI)

An industrial safety inspection web application designed to automatically detect Personal Protective Equipment (PPE) using state-of-the-art computer vision models.

---

## 🚀 Features & Capabilities / Características
- **Multiclass Detection:** Identifies Helmets (`helmet`), Vests (`vest`), Boots (`boots`), Gloves (`gloves`), and Goggles (`goggles`).
- **Gigapixel SAHI Mode:** Integrates *Slicing Aided Hyper Inference* to slice high-res photos and detect small accessories or distant workers accurately.
- **OHS Compliance Engine:** Spatial relationship logic between safety helmets and vests, featuring distance-based filtering to ignore background false alarms.
- **Bilingual Interface (i18n):** Instant toggle between Spanish and English directly from the sidebar.
- **CSV Audit Export:** Real-time generation and download of compliance reports.

---

## 🛠️ Tech Stack / Tecnologías
- **Core:** Python 3.10+, Streamlit
- **AI / Computer Vision:** Ultralytics YOLOv8, SAHI (Slicing Aided Hyper Inference)
- **Data & Image Processing:** OpenCV, Pillow, Pandas, NumPy

---

## 📦 Quick Start & Installation / Instalación

1. Clone the repository / Clona el repositorio:
   ```bash
   git clone [https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git](https://github.com/MarketablePlushiee/yolov8-safety-equipment-detection.git)
   cd yolov8-safety-equipment-detection
