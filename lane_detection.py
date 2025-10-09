from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("C:/Users/user/intellegent-control-week6/dataset-baru/dataset rel.v4i.yolov8/best.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result2.jpg")

# Contoh penggunaan
detect_rail_lane("rail.jpg")