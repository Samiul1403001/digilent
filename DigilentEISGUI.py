import sys, csv
import socket
import struct
import numpy as np
from PyQt6.QtWidgets import (QProgressBar, QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- CONFIGURATION ---
# The IP Address of your Digilent ADP3450
SERVER_IP = '10.115.78.182' 
SERVER_PORT = 5005

# --- WORKER THREAD (Handles Networking as CLIENT) ---
class NetworkThread(QThread):
    # Signals to communicate with the GUI (Main Thread)
    data_received = pyqtSignal(float, float, float, float) # Sends (Freq, Zreal, -Zimag)
    status_update = pyqtSignal(str, str)     # Sends (Color, Message)
    
    def __init__(self):
        super().__init__()
        self.sock = None # We will store the active socket here
        self.running = True

    def run(self):
        """This function runs in the background."""
        
        self.status_update.emit("orange", f"Connecting to {SERVER_IP}...")
        
        try:
            # Create a socket object
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5) # 5-second timeout for the initial connection
            
            # CONNECT to the Server (ADP3450)
            self.sock.connect((SERVER_IP, SERVER_PORT))
            self.sock.settimeout(None) # Remove timeout for the data loop
            
            self.status_update.emit("green", f"Connected to ADP3450!")
            
            # --- Data Reception Loop ---
            while self.running:
                # 1. Read Header (4 bytes) to know message size
                header = self.sock.recv(4)
                if not header: break # Server closed connection
                
                msg_len = struct.unpack('>I', header)[0]
                
                # 2. Read Payload (The float data)
                data_bytes = b''
                while len(data_bytes) < msg_len:
                    packet = self.sock.recv(msg_len - len(data_bytes))
                    if not packet: break
                    data_bytes += packet
                
                # 3. Convert bytes back to numpy array
                # Server sends [Freq, Z_real, Z_imag]
                eis_data = np.frombuffer(data_bytes, dtype=np.float64)
                
                # 4. Send Data to GUI
                # Check if we got the expected 3 values
                if len(eis_data) >= 5:
                    self.data_received.emit(10**eis_data[1], eis_data[2], eis_data[3], eis_data[6])
                
        except socket.timeout:
            self.status_update.emit("red", "Connection Timed Out.")
        except ConnectionRefusedError:
            self.status_update.emit("red", "Connection Refused (Is Server running?)")
        except Exception as e:
            self.status_update.emit("red", f"Error: {e}")
        finally:
            if self.sock: self.sock.close()

    def send_command(self, command_str):
        """Helper to send commands (START/STOP) to ADP3450"""
        if self.sock:
            try:
                self.sock.sendall(command_str.encode('utf-8'))
                print(f"Sent: {command_str}")
            except Exception as e:
                print(f"Send Error: {e}")

# --- MAIN WINDOW (GUI) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.recorded_data = [] 
        
        # Initialize UI first so window appears instantly
        self.initUI()
        
        # Start the Network Thread
        self.worker = NetworkThread()
        # Connect Signals from Thread -> GUI functions
        self.worker.data_received.connect(self.update_plot)
        self.worker.status_update.connect(self.update_status_display)
        self.worker.start()

    def initUI(self):
        self.setWindowTitle("ADP3450 Host Controller")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #F0F0F0;")

        # --- Styles ---
        btn_style = "QPushButton { font-size: 16px; font-weight: bold; border-radius: 8px; padding: 10px; color: white; }"
        start_style = btn_style + "QPushButton { background-color: #2E7D32; }"
        stop_style =  btn_style + "QPushButton { background-color: #C62828; }"
        save_style =  btn_style + "QPushButton { background-color: #1565C0; }"

        # --- Left Panel ---
        left_layout = QVBoxLayout()

        # Status Section
        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)
        self.status_light.setStyleSheet("background-color: grey; border-radius: 10px; border: 1px solid #555;")
        
        self.status_text = QLabel("Initializing...")
        self.status_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_light)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

        left_layout.addLayout(status_layout)
        left_layout.addSpacing(20)

        # Progress Bar Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50; /* Green Chunk */
                width: 10px;
            }
        """)
        
        left_layout.addWidget(QLabel("Sweep Progress:"))
        left_layout.addWidget(self.progress_bar)
        left_layout.addSpacing(20)

        # --- SoH Indicator ---
        self.lbl_soh = QLabel("Estimated SoH (progressive): ")
        self.lbl_soh.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: #333; 
            padding: 5px; 
            border: 1px solid #CCC; 
            background-color: white; 
            border-radius: 4px;
        """)
        left_layout.addWidget(self.lbl_soh)
        left_layout.addSpacing(20)

        # Buttons Section
        self.btn_start = QPushButton("START CAPTURE")
        self.btn_start.setStyleSheet(start_style)
        self.btn_start.clicked.connect(self.send_start)

        self.btn_stop = QPushButton("STOP CAPTURE")
        self.btn_stop.setStyleSheet(stop_style)
        self.btn_stop.clicked.connect(self.send_stop)
        self.btn_stop.setEnabled(False) 

        self.btn_save = QPushButton("SAVE DATA")
        self.btn_save.setStyleSheet(save_style)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_save.setEnabled(False)

        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_stop)
        left_layout.addWidget(self.btn_save)
        left_layout.addStretch() 
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(300)

        # --- Right Panel (SCATTER PLOT) ---
        self.figure = Figure(facecolor='#F0F0F0')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylabel("-Z_imag (\u03A9)")
        self.ax.set_xlabel("Z_real (\u03A9)")
        self.ax.set_title("Nyquist Plot")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.margins(0.001)
        
        self.scatter = self.ax.scatter([], [], c='blue', s=60, alpha=0.6, edgecolors='none')

        # --- Main Layout ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.canvas)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # --- SLOTS (Responding to Signals) ---
    @pyqtSlot(float, float, float, float)
    def update_plot(self, Freq, Z_real, Z_imag, SoH):
        """Updates Progress Bar and Plot"""
        
        # 1. Update Progress Bar (Logarithmic)
        try:
            start_log = np.log10(10)  # 3.0
            stop_log  = np.log10(0.011)  # -2.0
            span      = start_log - stop_log 
            current_log = np.log10(Freq)
            
            # Progress calculation (High -> Low sweep)
            fraction = (start_log - current_log) / span
            percent = np.clip(int(fraction * 100), 0, 100)
            self.progress_bar.setValue(max(0, min(100, percent)))

            # 3. Update the Text
            self.lbl_soh.setText(f"Estimated SoH (progressive): {SoH:.2f} %")
            
            # Optional: Change color if SoH is bad
            if SoH < 80:
                self.lbl_soh.setStyleSheet("font-size: 14px; font-weight: bold; color: red; background-color: #FFEBEE; border: 1px solid red; padding: 5px; border-radius: 4px;")
            else:
                self.lbl_soh.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E7D32; background-color: #E8F5E9; border: 1px solid #2E7D32; padding: 5px; border-radius: 4px;")
        except:
            self.lbl_soh.setText("SoH Error")
        
        # 2. Update Scatter Plot
        self.recorded_data.append((Freq, Z_real, Z_imag))
        
        # Keep only recent points to prevent lag if necessary (optional)
        # points_to_plot = self.recorded_data[-200:] 
        points_to_plot = self.recorded_data
        
        if len(points_to_plot) > 0:
            # Extract Z_real (Index 1) and Z_imag (Index 2)
            # We convert list of tuples -> numpy array
            arr = np.array(points_to_plot)
            # Scatter Plot X = Z_real, Y = -Z_imag (Nyquist Convention)
            xy_data = np.column_stack((arr[:, 1], arr[:, 2]))
            
            self.scatter.set_offsets(xy_data)

            # --- SMART ZOOM (Percentile Based) ---
            # Only switch to smart zoom if we have enough points (e.g., > 10)
            # Otherwise, standard min/max is safer for the start.
            if len(points_to_plot) > 2:
                # Get the 5th and 95th percentiles (ignores top/bottom 5% outliers)
                x_min = np.percentile(arr[:, 1], 5)
                x_max = np.percentile(arr[:, 1], 95)
                y_min = np.percentile(arr[:, 2], 10)
                y_max = np.percentile(arr[:, 2], 95)
            else:
                # Standard Min/Max for the first few points
                x_min, x_max = 0, arr[:, 1].max()
                y_min, y_max = arr[:, 2].min(), arr[:, 2].max()
            
            # Calculate Padding (10% of the visible range)
            x_span = x_max - x_min
            y_span = y_max - y_min
            
            # Prevent crash if all points are identical (span = 0)
            if x_span == 0: x_span = 1.0
            if y_span == 0: y_span = 1.0

            # Apply Limits with Padding
            self.ax.set_xlim(x_min + (x_span * 0.05), x_max + (x_span * 0.05))
            self.ax.set_ylim(y_min - (y_span * 0.1), y_max + (y_span * 0.1))

            self.canvas.draw()

    @pyqtSlot(str, str)
    def update_status_display(self, color, text):
        if color == "red": hex_c = "#FF5252"
        elif color == "orange": hex_c = "#FF9800"
        elif color == "green": hex_c = "#00E676"
        else: hex_c = "grey"
        
        self.status_light.setStyleSheet(f"background-color: {hex_c}; border-radius: 10px;")
        self.status_text.setText(text)

    # --- BUTTON HANDLERS ---
    def send_start(self):
        self.recorded_data = [] 
        self.scatter.set_offsets(np.empty((0, 2)))
        self.canvas.draw()
        
        self.worker.send_command("START")
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)

    def send_stop(self):
        self.worker.send_command("STOP")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)

    def save_data(self):
        if not self.recorded_data: return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")

        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Frequency (Hz)", "Z_real (Ohms)", " -Z_imag (Ohms)"])
                    writer.writerows(self.recorded_data)
                self.status_text.setText("Status: Data Saved!")
            except Exception as e:
                self.status_text.setText("Status: Error Saving")
                print(f"Error saving file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())