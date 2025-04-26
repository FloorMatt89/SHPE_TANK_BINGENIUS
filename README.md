Bingenius - Smart Waste Classifier
Prototype for SHPE TANK 2025
Theme: Sustainability | Duration: 3 months
Mentorship: Isa @ Accenture

Project Overview
This project aimed to design a Smart Waste Sorting System for sustainable waste management.
The core goal was to detect or classify recyclable versus non-recyclable items in real time using computer vision and machine learning on a Raspberry Pi 5 setup.

This prototype was developed as part of the SHPE TANK initiative at the University of Florida (UF), under a mentorship program guided by a UF alumnus currently working at Accenture.

Models Attempted
1. Object Detection Model
Dataset: Custom waste detection dataset from Roboflow.

Model: YOLOv8n-obb (Oriented Bounding Boxes).

Training Details:

Trained on Roboflow to do annotations.


Testing Accuracy: 73% mAP

Challenge:

Due to the heavy computational load of real-time object detection, performance on the Raspberry Pi 5 was too slow (~1 FPS).

As a result, we pivoted to a more efficient approach based on image classification.

2. Multiclass Image Classification Model
Dataset: Waste image dataset sourced from Kaggle (various recyclable and trash materials).

Model: Custom CNN-based classifier.

We developed a CNN-based multiclass image classifier using a transfer-learned ResNet50 backbone and fine-tuned it on a custom waste classification dataset

Training Details:

Trained and hyperparameter-tuned on HiPerGator.

Adjusted optimizers, batch size, learning rates to maximize generalization.

Testing Accuracy: 87%

Deployment:

Converted the trained model into ONNX format for lightweight deployment.

Real-time predictions were performed using Picamera2 and FastAPI hosted locally on the Raspberry Pi 5.

Servo motors were triggered based on the prediction (trash vs recyclable classification).

Key Technologies
Python, FastAPI, ONNX Runtime

TensorFlow/Keras (for classification model training)

Ultralytics YOLOv8 (for object detection experiments)

OpenCV (camera and visualization)

RPi.GPIO (motor control)

Picamera2 (Raspberry Pi camera module API)

HiPerGator (model training + tuning)

Final Prototype Demo
Functioning real-time waste classification on Raspberry Pi 5.

Servo motors sort detected items into "Recycle" or "Trash" bins based on classification.

Web interface (FastAPI MJPEG streaming) to view live camera feed and predictions.

Reliable speed and accuracy after switching to a classification-first approach.

Personal Contribution
Led the software development: dataset preprocessing, model training, ONNX conversion, Raspberry Pi deployment.

Assisted with the sales pitch during the SHPE TANK competition event.

Worked collaboratively with the team on prototyping and demonstration strategies.

Lessons Learned
The critical importance of hardware-aware model selection.

Trade-offs between accuracy and latency for real-world prototypes.

Gained hands-on experience integrating machine learning models into embedded systems.

Practical exposure to HiPerGator HPC resources for large model training.

Special Thanks
Big thank you to the Society of Hispanic Professional Engineers (SHPE) and our amazing mentor from Accenture for their support and guidance throughout this project!
