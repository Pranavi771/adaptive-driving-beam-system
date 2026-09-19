import os
# --- FIX: Force Python to look in the same folder as this script ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -------------------------------------------------------------------

import cv2
import numpy as np
import time
import requests 

# ================= CONFIGURATION =================
# --- UPDATE THESE IPs to match your ESP32s ---
CAMERA_IP   = "192.168.1.10"
LED_IP      = "192.168.1.11"
# -------------------------------------------------

STREAM_URL  = f"http://{CAMERA_IP}:81/stream"
COMMAND_URL = f"http://{LED_IP}/leds"

CONF_THRESHOLD = 0.5  
NMS_THRESHOLD  = 0.4   
YOLO_WIDTH     = 416      
YOLO_HEIGHT    = 416     
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

TARGET_CLASSES = ['car', 'person', 'truck', 'motorbike']
GRID_X = 144 
# =================================================

print(f"Current working directory: {os.getcwd()}")
print("Loading YOLOv4-tiny model...")
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    layer_names = net.getLayerNames()
    try:
        output_layer_indices = net.getUnconnectedOutLayers().flatten()
    except AttributeError:
        output_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    print("YOLO model loaded successfully!")
except FileNotFoundError as e:
    print(f"\nERROR: Could not find a file!")
    print(f"Missing file: {e.filename}")
    print("Please make sure coco.names, yolov4-tiny.cfg, and yolov4-tiny.weights are in:")
    print(f"{os.getcwd()}")
    exit()

print(f"Connecting to camera at {STREAM_URL}...")
cap = cv2.VideoCapture(STREAM_URL)
time.sleep(2.0) 
if not cap.isOpened():
    print("Error: Cannot connect to camera stream.")
    exit()
print("Stream connected. Starting main loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame lost, attempting reconnect...")
            cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(STREAM_URL)
            continue
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (YOLO_WIDTH, YOLO_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESHOLD and classes[class_id] in TARGET_CLASSES:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if isinstance(indices, np.ndarray): indices = indices.flatten()

        dimmed_leds = []
        segment_width = width // GRID_X
        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                center_x = box[0] + box[2] // 2
                segment_index = min(center_x // segment_width, GRID_X - 1)
                if segment_index not in dimmed_leds: dimmed_leds.append(segment_index)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

        try:
            dim_string = ",".join(map(str, dimmed_leds)) if dimmed_leds else "-1"
            requests.get(COMMAND_URL, params={'dim': dim_string}, timeout=0.05)
        except: pass 

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Adaptive Headlight", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()