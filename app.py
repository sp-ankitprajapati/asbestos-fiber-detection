"""Local Gradio demo for a trained fiber-mark detector."""
import argparse
from pathlib import Path

import numpy as np

from data_utils import validate_model_names


def run_detection(model, image_rgb, confidence, device=None):
    """Detect marks in one RGB image. Returns (annotated RGB image, table rows)."""
    # Gradio gives RGB arrays, but Ultralytics reads NumPy input as BGR (OpenCV order).
    # Without this flip the model would see red and blue swapped.
    image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
    result = model.predict(image_bgr, conf=confidence, iou=0.7, imgsz=640,
                           max_det=300, rect=False, device=device, verbose=False)[0]
    rows = [[result.names[int(cls)], round(float(score), 4),
             *[round(float(v), 1) for v in box]]
            for cls, score, box in zip(result.boxes.cls.tolist(),
                                       result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
    # plot() draws on the BGR input and returns BGR; flip back to RGB for display.
    return np.ascontiguousarray(result.plot()[..., ::-1]), rows


def build_demo(weights, device=None):
    import gradio as gr
    from ultralytics import YOLO

    model = YOLO(str(weights))
    validate_model_names(model)

    def detect(image, confidence):
        if image is None:
            raise gr.Error("Upload a microscope image first.")
        annotated, rows = run_detection(model, image, confidence, device)
        return annotated, rows, f"{len(rows)} marks above {confidence:.2f} confidence"

    return gr.Interface(
        fn=detect,
        inputs=[gr.Image(type="numpy", image_mode="RGB", label="Microscope image"),
                gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")],
        outputs=[gr.Image(type="numpy", label="Detected marks"),
                 gr.Dataframe(headers=["Class", "Confidence", "x1", "y1", "x2", "y2"],
                              interactive=False, label="Detections (pixel coordinates)"),
                 gr.Textbox(label="Summary")],
        title="Fiber Mark Detection",
        description="Detect thick/thin and dark/light marks in microscope images. "
                    "The labels describe visual appearance, not mineral type. "
                    "This prototype assists image review; it does not determine whether a material contains asbestos.",
        concurrency_limit=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.error(f"{args.weights} not found. Run python train.py or supply --weights with your trained checkpoint.")
    build_demo(args.weights, args.device).queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
