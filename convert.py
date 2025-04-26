import tensorflow as tf
import tf2onnx
import os


def export_model_to_onnx(h5_path="waste_classifier.h5", onnx_path="waste_classifier.onnx"):
    print(f"Loading trained model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)

    print("Converting to ONNX format...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    export_model_to_onnx()

