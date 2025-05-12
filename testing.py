import numpy as np
import tensorflow as tf
import cv2
import time
import os
import requests
from threading import Thread
import queue
import threading
from mqtt_client import MQTTClient

MODEL_URL = "https://raw.githubusercontent.com/agummds/Mask-RCNN-TA/master/model.tflite"
MODEL_PATH = "model.tflite"

# Fixed camera parameters (calibrated once)
CAMERA_FOV = 58.9  # Diagonal Field of view in degrees
CAMERA_FOCAL_LENGTH = 0.3  # Focal length in cm
SENSOR_WIDTH_CM = 0.35  # Sensor width in cm
FIXED_DISTANCE = 150  # Fixed distance in cm
CAMERA_RESOLUTION = 1920  # Camera horizontal resolution in pixels

# Frame settings
FRAME_WIDTH = 480
FRAME_HEIGHT = 360
TARGET_FPS = 10
FRAME_TIME = 1.0/TARGET_FPS

# Fixed thread count
OPTIMAL_THREADS = 4

# Inisialisasi MQTT client
mqtt_client = MQTTClient()
mqtt_client.connect()

def pixel_to_cm(pixel_length):
    """
    Convert pixel length to cm using the fixed ratio of 0.187 cm per pixel
    """
    return pixel_length * 0.187

def calculate_distance(pixel_height):
    """
    Calculate real-world distance based on pixel height using diagonal FOV.
    Using the formula: distance = (object_height * focal_length) / (2 * tan(FOV/2) * pixel_height)
    """
    # Convert diagonal FOV to horizontal FOV using aspect ratio
    aspect_ratio = FRAME_WIDTH / FRAME_HEIGHT
    horizontal_fov = 2 * np.arctan(np.tan(np.radians(CAMERA_FOV/2)) * aspect_ratio)
    horizontal_fov_degrees = np.degrees(horizontal_fov)
    
    # Calculate distance using horizontal FOV
    return (30.0 * CAMERA_FOCAL_LENGTH * FRAME_HEIGHT) / (pixel_height * 2 * np.tan(np.radians(horizontal_fov_degrees/2)))

def download_model_once():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GitHub...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded and saved successfully.")
            print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please check if the model URL is correct and accessible.")
            raise
    else:
        print(f"Model already exists at {MODEL_PATH}")
        print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

def load_model():
    print(f"Loading model with {OPTIMAL_THREADS} threads...")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=OPTIMAL_THREADS
        )
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        
        print(f"Model loaded successfully.")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def process_frame(frame, interpreter, input_details, output_details):
    """Process a single video frame"""
    original_image = frame.copy()
    
    # Resize image to match model input shape
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    image_resized = cv2.resize(frame, (input_width, input_height))
    
    # Normalize the image
    image_array = image_resized.astype(np.float32) / 255.0
    
    if len(input_details[0]['shape']) == 4:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get model outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])  # Confidence scores
    masks = interpreter.get_tensor(output_details[3]['index'])  # Instance masks
    
    # Process each detection
    for i in range(len(scores[0])):
        if scores[0][i] > 0.5:  # Confidence threshold
            box = boxes[0][i]
            class_id = int(classes[0][i])
            score = scores[0][i]
            mask = masks[0][i]
            
            # Convert box coordinates to image space
            x1, y1, x2, y2 = box
            x1 = int(x1 * original_image.shape[1])
            y1 = int(y1 * original_image.shape[0])
            x2 = int(x2 * original_image.shape[1])
            y2 = int(y2 * original_image.shape[0])
            
            # Draw bounding box with thicker lines
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red box, thickness 4
            
            # Draw mask with transparency
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask_color = np.zeros_like(original_image)
            mask_color[mask > 0] = [0, 255, 0]  # Green mask
            original_image = cv2.addWeighted(original_image, 0.7, mask_color, 0.3, 0)
            
            # Add label with background
            label = f"Class {class_id}: {score:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(original_image, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), (0, 0, 0), -1)
            cv2.putText(original_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate measurements
            height = y2 - y1
            width = x2 - x1
            height_cm = pixel_to_cm(height)
            width_cm = pixel_to_cm(width)
            
            # Kirim data pengukuran melalui MQTT
            mqtt_client.publish_measurement(
                height_cm=height_cm,
                width_cm=width_cm,
                confidence=float(score),
                class_id=class_id
            )
            
            # Add measurements with background
            measurements = f"H: {height_cm:.1f}cm W: {width_cm:.1f}cm"
            text_size = cv2.getTextSize(measurements, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(original_image, (x1, y2 + 5), 
                         (x1 + text_size[0], y2 + text_size[1] + 15), (0, 0, 0), -1)
            cv2.putText(original_image, measurements, (x1, y2 + text_size[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add performance info
    fps = 1.0 / inference_time if inference_time > 0 else 0
    info_text = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms"
    cv2.putText(original_image, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return original_image

def read_frames(cap, frameQueue, stopEvent):
    while not stopEvent.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
            
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        if frameQueue.qsize() > 1:
            try:
                frameQueue.get_nowait()
            except queue.Empty:
                pass
        frameQueue.put(frame)
        
        time.sleep(0.001)

def run_realtime_detection():
    try:
        download_model_once()
        print("Loading model...")
        interpreter = load_model()
        print("Model loaded successfully")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        # Initialize camera
        print("\nInitializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to open camera, trying alternative index...")
            cap = cv2.VideoCapture(1)
            
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized with:")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"FPS: {actual_fps}")
        
        # Test camera
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("Error: Could not read frame from camera")
            return
        print("Camera test successful - frame captured")
        
        # Setup frame queue and thread
        frameQueue = queue.Queue(maxsize=2)
        stopEvent = threading.Event()
        frameThread = Thread(target=read_frames, args=(cap, frameQueue, stopEvent))
        frameThread.daemon = True
        frameThread.start()

        print("\nCamera is ready!")
        print("Press 'q' to quit.")
        
        while True:
            try:
                if frameQueue.empty():
                    time.sleep(0.001)
                    continue
                    
                frame = frameQueue.get()
                
                # Process frame
                result = process_frame(frame, interpreter, input_details, output_details)
                
                # Show visualization
                cv2.imshow("Object Detection", result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'stopEvent' in locals():
            stopEvent.set()

if __name__ == "__main__":
    run_realtime_detection()