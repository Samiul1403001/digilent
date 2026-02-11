import sys, csv
import socket
import struct
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- WORKER THREAD (Handles Networking) ---
class NetworkThread(QThread):
    # Signals to communicate with the GUI (Main Thread)
    data_received = pyqtSignal(float, float) # Sends (Zreal, -Zimag)
    status_update = pyqtSignal(str, str)     # Sends (Color, Message)
    
    def __init__(self):
        super().__init__()
        self.conn = None # We will store the active connection here
        self.running = True

    def run(self):
        """This function runs in the background."""
        HOST = '0.0.0.0'
        PORT = 5005
        
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Allow reuse of address to avoid "Address already in use" errors
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_sock.bind((HOST, PORT))
            server_sock.listen(1)
            
            # Update GUI status
            self.status_update.emit("orange", "Waiting for ADP3450...")
            
            # This line BLOCKS, but since we are in a thread, the GUI is fine.
            self.conn, addr = server_sock.accept()
            
            self.status_update.emit("green", f"Connected to ADP3450!")
            
            # --- Data Reception Loop ---
            timestep = 0
            while self.running:
                # 1. Read Header (4 bytes) to know message size
                header = self.conn.recv(4)
                if not header: break # Connection closed
                
                msg_len = struct.unpack('>I', header)[0]
                
                # 2. Read Payload (The float data)
                # TCP stream might deliver data in chunks; we must ensure we get it all.
                data_bytes = b''
                while len(data_bytes) < msg_len:
                    packet = self.conn.recv(msg_len - len(data_bytes))
                    if not packet: break
                    data_bytes += packet
                    
                # 3. Convert bytes back to numpy array
                data_np = np.frombuffer(data_bytes, dtype=np.float64)
                eis_data = data_np.reshape(1, 3)
                
                # 4. Send Data to GUI
                self.data_received.emit(eis_data[:, 1], eis_data[:, 2])
                timestep += 1
                
        except Exception as e:
            self.status_update.emit("red", f"Error: {e}")
        finally:
            if self.conn: self.conn.close()
            server_sock.close()

    def send_command(self, command_str):
        """Helper to send data back to ADP3450"""
        if self.conn:
            try:
                self.conn.sendall(command_str.encode('utf-8'))
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
        self.setWindowTitle("ADP3450 powered EIS")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #F0F0F0;")

        # --- Styles ---
        btn_style = "QPushButton { font-size: 16px; font-weight: bold; border-radius: 8px; padding: 10px; color: white; }"
        start_style = btn_style + "QPushButton { background-color: #2E7D32; }"
        stop_style =  btn_style + "QPushButton { background-color: #C62828; }"
        save_style =  btn_style + "QPushButton { background-color: #1565C0; }"

        # --- Left Panel ---
        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)
        self.status_light.setStyleSheet("background-color: grey; border-radius: 10px; border: 1px solid #555;")
        
        self.status_text = QLabel("Initializing...")
        self.status_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_light)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

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

        left_layout = QVBoxLayout()
        left_layout.addLayout(status_layout)
        left_layout.addSpacing(20)
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
        self.ax.set_title("Live EIS Data")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        # Initialize Scatter Plot
        # s=50 sets the size of dots
        # alpha=0.6 makes them slightly transparent
        self.scatter = self.ax.scatter([], [], c='blue', s=50, alpha=0.6, edgecolors='none')

        # --- Main Layout ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.canvas)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # --- SLOTS (Responding to Signals) ---
    @pyqtSlot(float, float)
    def update_plot(self, X, Y):
        """Updates the scatter plot points."""
        self.recorded_data.append((X, Y))
        
        # Keep only last 100 points for performance
        recent = self.recorded_data[-65:]
        
        # Prepare (x, y) pairs for scatter
        # Scatter plots require a 2D array of [[x1, y1], [x2, y2], ...]
        points = np.array(recent)
        
        if len(points) > 0:
            self.scatter.set_offsets(points)
            
            # Dynamic coloring (Optional): Turn dots Red if > 0.5
            # colors = ['red' if y > 0.5 else 'blue' for y in points[:, 1]]
            # self.scatter.set_color(colors)
            
            # Autoscale view
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()

    @pyqtSlot(str, str)
    def update_status_display(self, color, text):
        """Called when connection status changes"""
        # Map color names to hex if needed, or use simple logic
        if color == "red": hex_c = "#FF5252"
        elif color == "orange": hex_c = "#FF9800"
        elif color == "green": hex_c = "#00E676"
        else: hex_c = "grey"
        
        self.status_light.setStyleSheet(f"background-color: {hex_c}; border-radius: 10px;")
        self.status_text.setText(text)

    # --- BUTTON HANDLERS ---
    def send_start(self):
        # 1. Clear the historical data list
        self.recorded_data = [] 
        
        # 2. Visually clear the scatter plot immediately
        # We set the offsets to an empty array (0 points)
        self.scatter.set_offsets(np.empty((0, 2)))
        self.canvas.draw()

        # We access the worker thread to send data
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
        """Saves the recorded_data list to a CSV file."""
        if not self.recorded_data:
            return

        # Open File Dialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv);;All Files (*)")

        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["TimeStep", "Voltage"]) # Header
                    writer.writerows(self.recorded_data) # Data
                self.status_text.setText("Status: Data Saved!")
            except Exception as e:
                self.status_text.setText("Status: Error Saving")
                print(f"Error saving file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())