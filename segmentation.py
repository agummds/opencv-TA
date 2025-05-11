import cv2
import numpy as np
import tensorflow as tf
import time
import os
import requests
from lcd_display import LCDDisplay
from flask import Flask, jsonify
import threading

# Flask app initialization
app = Flask(__name__)

# Global variable to store the latest measurement
latest_measurement = {
    'value': 0,
    'unit': 'cm'
}

# Constants
MODEL_URL = "https://raw.githubusercontent.com/agummds/Mask-RCNN-TA/master/model.tflite"
MODEL_PATH = "model.tflite"
FIXED_DISTANCE = 150  # cm
PIXEL_TO_CM = 0.187  # cm per pixel
MODEL_INPUT_SIZE = 640

# Flask route to get measurement
@app.route('/')
def get_measurement():
    return jsonify(latest_measurement)

def update_measurement(height_cm):
    """Update the latest measurement"""
    global latest_measurement
    latest_measurement = {
        'value': round(height_cm, 2),
        'unit': 'cm'
    }

# ... rest of your existing functions ...

def process_frame(frame, interpreter, input_details, output_details, lcd_display=None):
    """Process a single frame for body segmentation and measurement"""
    # ... your existing process_frame code ...
    
    if contours:
        # ... your existing contour processing code ...
        
        # Calculate measurements
        width_cm = w * PIXEL_TO_CM
        height_cm = h * PIXEL_TO_CM
        
        # Update the latest measurement for the Flask server
        update_measurement(height_cm)
        
        # ... rest of your existing process_frame code ...
    
    return result

def run_flask_server():
    """Run the Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=80)

def main():
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.daemon = True  # This ensures the thread will close when the main program exits
    flask_thread.start()
    
    # Download and load model
    if not download_model():
        return
    
    interpreter, input_details, output_details = load_model()
    if interpreter is None:
        return
    
    # Initialize LCD display
    try:
        lcd = LCDDisplay()
        print("LCD display initialized successfully")
    except Exception as e:
        print(f"Error initializing LCD display: {e}")
        lcd = None
    
    # Initialize camera with minimal settings
    print("Initializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Quick test frame
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return
    
    print("\nCamera initialized successfully")
    print("Press 'q' to quit")
    print("Server running at http://0.0.0.0:80")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        result = process_frame(frame, interpreter, input_details, output_details, lcd)
        
        # Show result
        cv2.imshow("Segmentation and Measurement", result)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if lcd is not None:
        lcd.cleanup()

if __name__ == "__main__":
    main() 