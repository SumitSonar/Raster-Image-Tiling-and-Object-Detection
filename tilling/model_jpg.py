# Import necessary libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model 
model = YOLO('E:/GIS/defence/detection_project/yolov8/yolov8m.pt')

# Function to perform detection on an image
def detect_objects(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model.predict(source=image_path, conf=0.1)  # Adjust confidence threshold as needed
    # Display results
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the bounding box
            conf = box.conf[0]  # Get confidence score
            cls = int(box.cls[0])  # Get class label
            
            # Draw bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'Class: {cls}, Conf: {conf:.2f}', (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Show the image with detections
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()

# Test the function with an example image path
detect_objects('D:/Synthetic_train/train/images/Zhukovsky_Airport_0_0_182_51838.png')  # Replace with your image path

# E:\GIS\defence\detection_project\tiles\tile_159.png
# D:/Synthetic_train/train/images/Atlanta_Airport_0_0_1_917.png
# D:/S3 bucket/synthetic/test/images/Munich_Airport_0_0_480_30161.png
# D:/Synthetic_train/train/images/Zhukovsky_Airport_0_0_182_51838.png