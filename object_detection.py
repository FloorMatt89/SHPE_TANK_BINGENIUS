import os
from ultralytics import YOLO


def train():
    dataset_path = os.path.abspath("waste-obb/data.yaml")
    model = YOLO("yolov8s-obb.pt")  

    results = model.train(
        data=dataset_path,  
        epochs=25,
        imgsz=640,
        batch=8,
        name="waste_yolo_obb"
    )

    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
