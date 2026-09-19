import cv2
import numpy as np
import time
import sys  # To read command-line arguments

# --- NEW NATIVE PI CAMERA LIBRARY ---
from picamera2 import Picamera2

# --- LED LIBRARY ---
from rpi_ws281x import *

# --- CONFIGURATION ---
# YOLO Settings
CONF_THRESHOLD = 0.5  # Confidence threshold
NMS_THRESHOLD = 0.4   # Non-Maximum Suppression
YOLO_WIDTH = 416      # MUST MATCH .cfg FILE
YOLO_HEIGHT = 416     # MUST MATCH .cfg FILE

# --- VIDEO CAPTURE SIZE ---
# Smaller is faster for the Pi to process
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Objects to detect and dim for
TARGET_CLASSES = ['car', 'person', 'truck', 'motorbike']

# --- LED STRIP CONFIG ---
LED_COUNT      = 144             # Number of LED pixels.
LED_PIN        = 18              # GPIO pin (18 is PWM).
LED_FREQ_HZ    = 800000          # LED signal frequency
LED_DMA        = 10              # DMA channel
LED_BRIGHTNESS = 150             # 0-255
LED_INVERT     = False           # True to invert the signal
LED_CHANNEL    = 0               

# Define your colors
COLOR_BRIGHT = Color(255, 255, 255) # White
COLOR_DIM    = Color(30, 30, 30)    # Dim White

# Headlight Matrix Grid
GRID_X = 144 # 144 segments horizontally
GRID_Y = 1   # 1 segment vertically
NUM_SEGMENTS = GRID_X * GRID_Y # This will be 144
# --- END CONFIGURATION ---

# --- DEBUGGING FLAG ---
DEBUG_MODE = False
if "--debug" in sys.argv:
    DEBUG_MODE = True
    print("--- DEBUG MODE ON: Printing LED status (this will be slower) ---")
# --- END NEW ---


# --- 1. INITIALIZE LED STRIP ---
try:
    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, COLOR_BRIGHT)
    strip.show()
    print("LED strip initialized. Press Ctrl+C to stop.")
except Exception as e:
    print(f"Error initializing LED strip: {e}")
    print("Ensure you are running this with 'sudo'")
    exit()

# --- 2. LOAD YOLOv4-tiny MODEL (Corrected Section) ---
print("Loading YOLOv4-tiny model...")
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # --- THIS IS THE FIX ---
    layer_names = net.getLayerNames()
    try:
        # For newer OpenCV versions
        output_layer_indices = net.getUnconnectedOutLayers().flatten()
    except AttributeError:
        # For older OpenCV versions
        output_layer_indices = net.getUnconnectedOutLayers()
    
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    # --- END OF FIX ---

    print("YOLO model loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find model files.")
    print("Make sure 'yolov4-tiny.weights', 'yolov4-tiny.cfg', and 'coco.names' are in the same directory.")
    exit()
# --- END SECTION 2 ---


# --- 3. INITIALIZE PI CAMERA ---
print(f"Initializing Pi Camera at {CAM_WIDTH}x{CAM_HEIGHT}...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT)}, controls={"FrameRate": 25})
picam2.configure(config)
picam2.start()
time.sleep(2.0) # Give camera time to warm up
print("Camera initialized.")

# --- 4. MAIN PROCESSING LOOP ---
try:
    while True:
        # --- Capture Frame ---
        frame = picam2.capture_array()

        # --- THIS IS THE NEW FIX FOR THE CV2 ERROR ---
        # Convert 4-channel RGBA from camera to 3-channel BGR for YOLO
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        # --- END FIX ---
        
        # --- 4b. Object Detection ---
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (YOLO_WIDTH, YOLO_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESHOLD:
                    class_name = classes[class_id]
                    if class_name in TARGET_CLASSES:
                        center_x = int(detection[0] * CAM_WIDTH)
                        center_y = int(detection[1] * CAM_HEIGHT)
                        w = int(detection[2] * CAM_WIDTH)
                        h = int(detection[3] * CAM_HEIGHT)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        # --- 4c. Matrix Logic ---
        dimmed_led_indices = [] # Create a list to store dimmed LEDs
        
        # Set all LEDs to BRIGHT first
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, COLOR_BRIGHT)

        segment_width = CAM_WIDTH // GRID_X 
        segment_height = CAM_HEIGHT // GRID_Y 

        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                x = box[0]
                w = box[2]
                center_x = x + w // 2
                
                col = min(center_x // segment_width, GRID_X - 1) 
                segment_index = col # Since GRID_Y is 1

                # Set detected segments to DIM
                if 0 <= segment_index < strip.numPixels():
                    strip.setPixelColor(segment_index, COLOR_DIM)
                    
                    if DEBUG_MODE:
                        dimmed_led_indices.append(segment_index)

        # --- 4e. Send Data to LED Strip ---
        strip.show()
        
        # --- Conditional Print ---
        if DEBUG_MODE:
            if len(dimmed_led_indices) > 0:
                print(f"Dimming LEDs: {dimmed_led_indices}")
            else:
                print("No objects detected. All LEDs BRIGHT.")
        # --- END ---

# --- 5. CLEANUP ---
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    picam2.stop()
    # Turn all LEDs off on exit
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()
    print("Cleanup complete.")