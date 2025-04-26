import io
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import numpy as np
import onnxruntime as ort
import cv2
from picamera2 import Picamera2

app = FastAPI()

# --- Global camera state ---
camera_active = False

# --- Class Definitions ---
class_names = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes", "cardboard_packaging",
    "clothing", "coffee_grounds", "disposable_plastic_cutlery", "eggshells", "food_waste",
    "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars", "magazines", "newspaper",
    "office_paper", "paper_cups", "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags", "plastic_water_bottles",
    "shoes", "steel_food_cans", "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

RECYCLABLE_CLASSES = {
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes", "cardboard_packaging",
    "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars", "magazines", "newspaper",
    "office_paper", "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_water_bottles", "steel_food_cans"
}

RECYCLABLE_CATEGORY_MAP = {
    "aerosol_cans": "Metal",
    "aluminum_food_cans": "Metal",
    "aluminum_soda_cans": "Metal",
    "cardboard_boxes": "Cardboard",
    "cardboard_packaging": "Cardboard",
    "glass_beverage_bottles": "Glass",
    "glass_cosmetic_containers": "Glass",
    "glass_food_jars": "Glass",
    "magazines": "Paper",
    "newspaper": "Paper",
    "office_paper": "Paper",
    "plastic_cup_lids": "Plastic",
    "plastic_detergent_bottles": "Plastic",
    "plastic_food_containers": "Plastic",
    "plastic_shopping_bags": "Plastic",
    "plastic_soda_bottles": "Plastic",
    "plastic_water_bottles": "Plastic",
    "steel_food_cans": "Metal"
}

def is_recyclable(class_name: str) -> bool:
    return class_name in RECYCLABLE_CLASSES

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)
    frame = frame / 255.0
    frame = (frame - 0.5) / 0.5
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension -> (1, 224, 224, 3)
    return frame.astype(np.float32)


@app.get("/")
def homepage():
    return HTMLResponse(f"""
    <html>
        <head><title>Camera Stream</title></head>
        <body style="text-align: center;">
            <h1>Live Waste Classification </h1>
            <div>
                <button onclick="fetch('/start', {{method: 'POST'}})">Start Camera</button>
                <button onclick="fetch('/stop', {{method: 'POST'}})">Stop Camera</button>
            </div>
            <p id="status">Camera is currently: {"ON" if camera_active else "OFF"}</p>
            <img id="stream" src="/mjpeg" style="width: 80%; border: 2px solid black;" />
            <script>
                setInterval(() => {{
                    fetch("/status").then(res => res.text()).then(status => {{
                        document.getElementById("status").textContent = "Camera is currently: " + status;
                    }});
                }}, 1000);
            </script>
        </body>
    </html>
    """)

@app.get("/status")
def get_status():
    return "ON" if camera_active else "OFF"

@app.post("/start")
def start_stream():
    global camera_active
    camera_active = True
    return {"message": "Camera started"}

@app.post("/stop")
def stop_stream():
    global camera_active
    camera_active = False
    return {"message": "Camera stopped"}

@app.get("/mjpeg")
async def mjpeg_stream():
    global camera_active
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.set_controls({"AwbEnable": False, "ColourGains": (1.5, 1.5)})
    ort_session = ort.InferenceSession("waste_classifier.onnx", providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    def generate_frames():
        try:
            picam2.start()
            while camera_active:
                frame = picam2.capture_array()
                try:
                    input_tensor = preprocess(frame)
                    input_name = ort_session.get_inputs()[0].name
                    print(f"ONNX input shape: {input_tensor.shape}")

                    outputs = ort_session.run(None, {input_name: input_tensor})
                    print("ONNX outputs:", outputs)  
                    print("Output shape:", outputs[0].shape)
                    if not outputs or outputs[0] is None or len(outputs[0][0]) != len(class_names):
                        raise ValueError("Invalid ONNX model output")
                    pred_idx = np.argmax(outputs[0])
                except Exception as e:
                    logging.error(f"ONNX inference error: {e}")
                    print("This is the error")
                    continue  

                pred_class = class_names[pred_idx]
                recyclable = is_recyclable(pred_class)

                if recyclable:
                    category = RECYCLABLE_CATEGORY_MAP.get(pred_class, "Unknown")
                    label = f"Recyclable ({category})"
                    color = (0, 255, 0)
                else:
                    label = f"{pred_class} - Trash"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                _, jpeg = cv2.imencode(".jpg", rgb_frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception as e:
            logging.error(f"Camera stream error: {e}")
        finally:
            picam2.stop()
            picam2.close()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
