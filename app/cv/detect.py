from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

image_path = "images/test.jpg"

results = model(image_path)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

results[0].save(filename=str(output_dir / "result.jpg"))

print("Detection completed!")
print(f"Input image: {image_path}")
print("Output image saved to outputs/result.jpg")
