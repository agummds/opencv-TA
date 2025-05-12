import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

class MQTTClient:
    def __init__(self, broker="broker.hivemq.com", port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        # Topic untuk pengukuran
        self.measurement_topic = "raspberry/measurement"
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")
            
    def on_disconnect(self, client, userdata, rc):
        print(f"Disconnected with result code: {rc}")
        
    def connect(self):
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        
    def publish_measurement(self, height_cm, width_cm, confidence, class_id):
        """
        Publish hasil pengukuran ke MQTT broker
        Parameters:
        - height_cm: tinggi objek dalam cm
        - width_cm: lebar objek dalam cm
        - confidence: tingkat kepercayaan deteksi (0-1)
        - class_id: ID kelas objek yang terdeteksi
        """
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "measurement": {
                    "height": height_cm,
                    "width": width_cm,
                    "confidence": confidence,
                    "class_id": class_id,
                    "unit": "cm"
                }
            }
            self.client.publish(self.measurement_topic, json.dumps(payload))
        except Exception as e:
            print(f"Error publishing measurement data: {e}")

# Contoh penggunaan
if __name__ == "__main__":
    mqtt_client = MQTTClient()
    mqtt_client.connect()
    
    try:
        while True:
            # Di sini Anda bisa memanggil publish_measurement dengan data aktual
            # dari hasil deteksi objek
            time.sleep(5)  # Tunggu 5 detik
    except KeyboardInterrupt:
        print("Stopping MQTT client...")
        mqtt_client.disconnect() 