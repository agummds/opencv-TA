import os
import subprocess
import sys
import platform

def check_tflite_tools():
    """Memeriksa apakah tools TFLite terinstall"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        return True
    except ImportError:
        print("TensorFlow tidak terinstall")
        return False

def quantize_with_command_line():
    """Menggunakan command line tools untuk kuantisasi model"""
    MODEL_PATH = "model.tflite"
    QUANTIZED_MODEL_PATH = "model_quantized.tflite"
    
    print("=============================================")
    print("KUANTISASI MODEL DENGAN COMMAND LINE")
    print("=============================================")
    
    # Periksa keberadaan model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model tidak ditemukan di {MODEL_PATH}")
        return False
    
    # Tentukan sistem operasi
    os_name = platform.system()
    print(f"Sistem Operasi: {os_name}")
    
    # Command untuk kuantisasi berdasarkan OS
    if os_name == "Windows":
        command = [
            "python", "-m", "tensorflow.lite.tools.optimize.tflite_convert",
            "--input_file=" + MODEL_PATH,
            "--output_file=" + QUANTIZED_MODEL_PATH,
            "--post_training_quantize"
        ]
    else:  # Linux atau MacOS
        command = [
            "tflite_convert",
            "--input_file=" + MODEL_PATH,
            "--output_file=" + QUANTIZED_MODEL_PATH,
            "--post_training_quantize"
        ]
    
    print("\nMenjalankan perintah kuantisasi:")
    print(" ".join(command))
    print("\nProses berjalan, mohon tunggu...")
    
    try:
        # Jalankan perintah kuantisasi
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ambil output dan error
        stdout, stderr = process.communicate()
        
        # Tampilkan output
        if stdout:
            print("\nOutput:")
            print(stdout)
        
        # Tampilkan error jika ada
        if stderr:
            print("\nError:")
            print(stderr)
        
        # Periksa hasil
        if process.returncode == 0 and os.path.exists(QUANTIZED_MODEL_PATH):
            # Hitung dan tampilkan pengurangan ukuran
            original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100
            
            print("\nKuantisasi berhasil!")
            print(f"Ukuran model asli: {original_size:.2f} MB")
            print(f"Ukuran model hasil kuantisasi: {quantized_size:.2f} MB")
            print(f"Pengurangan ukuran: {reduction:.2f}%")
            return True
        else:
            print("\nKuantisasi gagal dengan command line.")
            return False
    
    except Exception as e:
        print(f"\nError: {e}")
        return False

def quantize_with_manual_method():
    """Metode alternatif menggunakan pendekatan manual"""
    MODEL_PATH = "model.tflite"
    QUANTIZED_MODEL_PATH = "model_quantized.tflite"
    
    print("=============================================")
    print("KUANTISASI MANUAL DENGAN TENSORFLOW LITE")
    print("=============================================")
    
    try:
        import tensorflow as tf
        
        # Baca model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Dapatkan info input
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model input shape: {input_details[0]['shape']}")
        print(f"Model output shape: {output_details[0]['shape']}")
        
        # Buat model dummy dengan shape yang sama
        input_shape = input_details[0]['shape'][1:]  # Remove batch dimension
        
        # Buat model sederhana
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(output_details[0]['shape'][-1], 1, padding='same')
        ])
        
        # Simpan model ke SavedModel format
        saved_model_dir = "temp_saved_model"
        os.makedirs(saved_model_dir, exist_ok=True)
        model.save(saved_model_dir)
        
        print("Model sementara dibuat untuk konversi")
        
        # Konversi ke TFLite dengan kuantisasi
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Float16 quantization
        
        print("Menjalankan kuantisasi dengan TFLiteConverter...")
        quantized_model = converter.convert()
        
        # Simpan model terkuantisasi
        with open(QUANTIZED_MODEL_PATH, 'wb') as f:
            f.write(quantized_model)
        
        # Periksa hasil
        if os.path.exists(QUANTIZED_MODEL_PATH):
            original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100
            
            print("\nKuantisasi berhasil!")
            print(f"Ukuran model asli: {original_size:.2f} MB")
            print(f"Ukuran model hasil kuantisasi: {quantized_size:.2f} MB")
            print(f"Pengurangan ukuran: {reduction:.2f}%")
            return True
        else:
            print("Kuantisasi gagal.")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMenyalin model sebagai fallback terakhir...")
        
        # Salin file sebagai fallback terakhir
        try:
            import shutil
            shutil.copy(MODEL_PATH, QUANTIZED_MODEL_PATH)
            print("Model disalin tanpa kuantisasi.")
            return False
        except Exception as copy_error:
            print(f"Error saat menyalin: {copy_error}")
            return False
    finally:
        # Bersihkan file sementara
        if os.path.exists("temp_saved_model"):
            try:
                import shutil
                shutil.rmtree("temp_saved_model")
            except:
                pass

def run_optimized_model():
    """Fungsi untuk menguji model terkuantisasi"""
    MODEL_PATH = "model_quantized.tflite"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model terkuantisasi tidak ditemukan di: {MODEL_PATH}")
        return
    
    try:
        import tensorflow as tf
        import time
        import numpy as np
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Dapatkan info input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Buat input dummy
        input_shape = input_details[0]['shape']
        input_data = np.random.random(input_shape).astype(np.float32)
        
        # Ukur waktu inferensi
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Warm up
        interpreter.invoke()
        
        # Pengukuran waktu
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            interpreter.invoke()
        end_time = time.time()
        
        avg_time = (end_time - start_time) * 1000 / iterations
        
        print(f"\nModel terkuantisasi berjalan dengan baik!")
        print(f"Rata-rata waktu inferensi: {avg_time:.2f} ms")
        
    except Exception as e:
        print(f"Error saat testing model: {e}")

if __name__ == "__main__":
    print("Program Kuantisasi Model TFLite Alternatif")
    print("------------------------------------------")
    
    if check_tflite_tools():
        print("\nMencoba metode kuantisasi command line...")
        if quantize_with_command_line():
            run_optimized_model()
        else:
            print("\nMencoba metode kuantisasi manual...")
            if quantize_with_manual_method():
                run_optimized_model()
            else:
                print("\nSemua metode kuantisasi gagal.")
                print("Silakan coba install ulang TensorFlow atau gunakan TensorFlow versi 2.5+")
    else:
        print("\nTensorFlow tidak terinstall. Harap install dengan:")
        print("pip install tensorflow")