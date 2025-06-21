# Bingenius – Smart Waste Classifier

**Prototype for SHPE TANK 2025**  
**Theme:** Sustainability  
**Duration:** 3 Months  
**Mentorship:** Isa @ Accenture

---

## Project Overview

Bingenius is a **Smart Waste Sorting System** developed for sustainable waste management using real-time computer vision and machine learning. The goal was to classify recyclable vs. non-recyclable items using a **Raspberry Pi 5**.

This prototype was built as part of the **SHPE TANK initiative at the University of Florida (UF)**, under mentorship from a UF alumnus now at **Accenture**.

---

## Models Attempted

### 1. Object Detection (YOLOv8n-OBB)

- **Dataset:** Custom-labeled waste detection dataset from Roboflow  
- **Model:** YOLOv8n with Oriented Bounding Boxes  
- **Training:** Annotated and trained on Roboflow  
- **Accuracy:** ~73% mAP  
- **Challenge:**  
  - Too computationally intensive for real-time use on Raspberry Pi 5  
  - Performance limited to ~1 FPS

 **Pivoted to a more efficient image classification approach**

---

### 2. Multiclass Image Classification (ResNet50)

- **Dataset:** Mixed recyclable/trash image dataset from Kaggle  
- **Model:** Transfer-learned ResNet50 with custom CNN layers  
- **Training:**  
  - Performed on **HiPerGator** HPC cluster  
  - Fine-tuned hyperparameters (optimizer, batch size, learning rate)  
- **Accuracy:** ~87%  
- **Deployment:**  
  - Converted to **ONNX** for lightweight inference  
  - Real-time predictions with **Picamera2** + **FastAPI**  
  - **Servo motors** triggered based on classification (Recycle or Trash)

---

## Key Technologies

- `Python`, `FastAPI`, `ONNX Runtime`  
- `TensorFlow/Keras` – model training  
- `Ultralytics YOLOv8` – object detection experiments  
- `OpenCV` – camera input & visualization  
- `RPi.GPIO` – motor control  
- `Picamera2` – Raspberry Pi camera API  
- `HiPerGator` – model training & tuning

---

## Final Prototype Demo

- Real-time waste classification running on Raspberry Pi 5  
- Servo motors sort items into **Recycle** or **Trash** bins  
- Web UI using **FastAPI MJPEG stream** for live predictions  
- Stable speed & accuracy with classification-based approach

---

##  Personal Contribution

- Led software development: data preprocessing, training, ONNX conversion, deployment  
- Assisted with SHPE TANK pitch  
- Collaborated on prototyping and demonstration strategies

---

##  Lessons Learned

- Importance of **hardware-aware model design**  
- Navigating **accuracy vs. latency** trade-offs in embedded ML  
- Gained real-world experience with **edge AI + deployment**  
- Leveraged **HiPerGator** for scalable model development

---

##  Special Thanks

Huge thanks to the **Society of Hispanic Professional Engineers (SHPE)** and our mentor **Isa from Accenture** for their continued support and guidance!

---

