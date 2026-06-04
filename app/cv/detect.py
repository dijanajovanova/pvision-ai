from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO("yolov8n.pt")

# Image path
image_path = "images/test.jpg"

# Run detection
results = model(image_path)

# Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Save annotated image
results[0].save(filename=str(output_dir / "result.jpg"))

print("Detection completed!")
print(f"Input image: {image_path}")
print("Output image saved to outputs/result.jpg")