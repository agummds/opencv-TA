from RPLCD.i2c import CharLCD
import time

class LCDDisplay:
    def __init__(self, i2c_address=0x27, i2c_port=1):
        """
        Initialize LCD display
        :param i2c_address: I2C address of the LCD (default: 0x27)
        :param i2c_port: I2C port number (default: 1 for Raspberry Pi)
        """
        try:
            self.lcd = CharLCD(i2c_expander='PCF8574',
                              address=i2c_address,
                              port=i2c_port,
                              cols=16,
                              rows=2,
                              dotsize=8)
            self.lcd.clear()
            self.lcd.write_string('System Ready')
            time.sleep(2)
            self.lcd.clear()
        except Exception as e:
            print(f"Error initializing LCD: {e}")
            self.lcd = None

    def display_measurements(self, width_cm, height_cm):
        """
        Display body measurements on LCD
        :param width_cm: Width in centimeters
        :param height_cm: Height in centimeters
        """
        if self.lcd is None:
            print("LCD not initialized")
            return

        try:
            self.lcd.clear()
            # Display width
            self.lcd.write_string('Width:')
            self.lcd.cursor_pos = (0, 7)
            self.lcd.write_string(f'{width_cm:.1f}cm')
            
            # Display height
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string('Height:')
            self.lcd.cursor_pos = (1, 7)
            self.lcd.write_string(f'{height_cm:.1f}cm')
        except Exception as e:
            print(f"Error displaying measurements: {e}")

    def clear(self):
        """Clear the LCD display"""
        if self.lcd is not None:
            self.lcd.clear()

    def cleanup(self):
        """Clean up LCD resources"""
        if self.lcd is not None:
            self.lcd.clear()
            self.lcd.close() 