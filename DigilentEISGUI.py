import sys, csv
import socket
import struct
import numpy as np
from PyQt6.QtWidgets import (QProgressBar, QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QLabel, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- CONFIGURATION ---
SERVER_IP = '10.115.78.182' 
SERVER_PORT = 5005

# --- WORKER THREAD (Handles Networking as CLIENT) ---
class NetworkThread(QThread):
    # Update signal to send a single numpy array containing all 21 elements
    data_received = pyqtSignal(np.ndarray) 
    status_update = pyqtSignal(str, str)     
    
    def __init__(self):
        super().__init__()
        self.sock = None 
        self.running = True

    def run(self):
        self.status_update.emit("orange", f"Connecting to {SERVER_IP}...")
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5) 
            self.sock.connect((SERVER_IP, SERVER_PORT))
            self.sock.settimeout(None) 
            
            self.status_update.emit("green", f"Connected to ADP3450!")
            
            while self.running:
                header = self.sock.recv(4)
                if not header: break 
                
                msg_len = struct.unpack('>I', header)[0]
                
                data_bytes = b''
                while len(data_bytes) < msg_len:
                    packet = self.sock.recv(msg_len - len(data_bytes))
                    if not packet: break
                    data_bytes += packet
                
                # Convert the 21-element bytes array back to numpy array
                eis_data = np.frombuffer(data_bytes, dtype=np.float64)
                
                # Check if we got all data for 3 cells (3 * 7 = 21 elements)
                if len(eis_data) >= 21:
                    self.data_received.emit(eis_data)
                
        except socket.timeout:
            self.status_update.emit("red", "Connection Timed Out.")
        except ConnectionRefusedError:
            self.status_update.emit("red", "Connection Refused (Is Server running?)")
        except Exception as e:
            self.status_update.emit("red", f"Error: {e}")
        finally:
            if self.sock: self.sock.close()

    def send_command(self, command_str):
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
        
        # Store data separately for the 3 cells
        self.data_c1 = [] 
        self.data_c2 = []
        self.data_c3 = []
        
        self.initUI()
        
        self.worker = NetworkThread()
        self.worker.data_received.connect(self.update_plot)
        self.worker.status_update.connect(self.update_status_display)
        self.worker.start()

    def create_plot_widget(self, title):
        """Helper to create a standard Nyquist plot widget"""
        figure = Figure(facecolor='#F0F0F0')
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        ax.set_ylabel("-Z_imag (\u03A9)")
        ax.set_xlabel("Z_real (\u03A9)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        scatter = ax.scatter([], [], c='blue', s=60, alpha=0.6, edgecolors='none')
        return canvas, ax, scatter
    
    def create_btn_style(self, base_color, hover_color, pressed_color):
            return f"""
                QPushButton {{ 
                    font-size: 16px; 
                    font-weight: bold; 
                    border-radius: 8px; 
                    padding: 12px; 
                    color: white; 
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    background-color: {base_color}; 
                }}
                QPushButton:hover {{ 
                    background-color: {hover_color}; 
                }}
                QPushButton:pressed {{ 
                    background-color: {pressed_color}; 
                }}
                QPushButton:disabled {{
                    background-color: #CFD8DC; 
                    color: #90A4AE;
                    border: none;
                }}
            """

    def initUI(self):
        self.setWindowTitle("ADP3450 Tri-Cell Analyzer")
        self.resize(1400, 900)
        self.setStyleSheet("background-color: #F0F0F0;")

        # --- 1. Control Panel (Bottom Right Corner) ---
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(20, 20, 20, 20)
        
        # Status
        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)
        self.status_light.setStyleSheet("background-color: grey; border-radius: 10px; border: 1px solid #555;")
        self.status_text = QLabel("Initializing...")
        self.status_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_light)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        control_layout.addLayout(status_layout)
        control_layout.addSpacing(15)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; height: 25px; }
            QProgressBar::chunk { background-color: #4CAF50; width: 10px; }
        """)
        control_layout.addWidget(QLabel("Global Sweep Progress:"))
        control_layout.addWidget(self.progress_bar)
        control_layout.addSpacing(15)

        # 3 SoH Indicators
        self.lbl_soh_c1 = QLabel("Cell 1 SoH: -- %")
        self.lbl_soh_c2 = QLabel("Cell 2 SoH: -- %")
        self.lbl_soh_c3 = QLabel("Cell 3 SoH: -- %")
        
        base_soh_style = "font-size: 14px; font-weight: bold; color: #333; padding: 5px; border: 1px solid #CCC; background-color: white; border-radius: 4px;"
        self.lbl_soh_c1.setStyleSheet(base_soh_style)
        self.lbl_soh_c2.setStyleSheet(base_soh_style)
        self.lbl_soh_c3.setStyleSheet(base_soh_style)
        
        control_layout.addWidget(self.lbl_soh_c1)
        control_layout.addWidget(self.lbl_soh_c2)
        control_layout.addWidget(self.lbl_soh_c3)
        control_layout.addSpacing(20)

        # Buttons
        # Generate the specific styles using the helper function
        style_start = self.create_btn_style("#4CAF50", "#43A047", "#2E7D32") # Green
        style_stop = self.create_btn_style("#F44336", "#E53935", "#C62828")  # Red
        style_save = self.create_btn_style("#2196F3", "#1E88E5", "#1565C0")  # Blue

        # Create and Apply to Buttons
        self.btn_start = QPushButton("START CAPTURE")
        self.btn_start.setStyleSheet(style_start)
        self.btn_start.clicked.connect(self.send_start)

        self.btn_stop = QPushButton("STOP CAPTURE")
        self.btn_stop.setStyleSheet(style_stop)
        self.btn_stop.clicked.connect(self.send_stop)
        self.btn_stop.setEnabled(False) 

        self.btn_save = QPushButton("SAVE ALL DATA")
        self.btn_save.setStyleSheet(style_save)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_save.setEnabled(False)

        # Add to Layout
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_save)
        control_layout.addStretch()
        
        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        # --- 2. Build the 2x2 Grid Layout ---
        grid = QGridLayout()
        
        # Create the 3 plots
        self.canvas1, self.ax1, self.scatter1 = self.create_plot_widget("Cell 1 Nyquist")
        self.canvas2, self.ax2, self.scatter2 = self.create_plot_widget("Cell 2 Nyquist")
        self.canvas3, self.ax3, self.scatter3 = self.create_plot_widget("Cell 3 Nyquist")

        # Top-Left: Cell 1
        grid.addWidget(self.canvas1, 0, 0)
        # Top-Right: Cell 2
        grid.addWidget(self.canvas2, 0, 1)
        # Bottom-Left: Cell 3
        grid.addWidget(self.canvas3, 1, 0)
        # Bottom-Right: Controls
        grid.addWidget(control_widget, 1, 1)

        central_widget = QWidget()
        central_widget.setLayout(grid)
        self.setCentralWidget(central_widget)

    def style_soh_label(self, label_widget, soh_val, prefix):
        """Helper to update SoH text and color dynamically"""
        label_widget.setText(f"{prefix}: {soh_val:.2f} %")
        if soh_val < 80:
            label_widget.setStyleSheet("font-size: 14px; font-weight: bold; color: red; background-color: #FFEBEE; border: 1px solid red; padding: 5px; border-radius: 4px;")
        else:
            label_widget.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E7D32; background-color: #E8F5E9; border: 1px solid #2E7D32; padding: 5px; border-radius: 4px;")

    def apply_smart_zoom(self, ax, data_list):
        """Helper to apply percentile zooming to a specific axis"""
        if len(data_list) > 2:
            arr = np.array(data_list)
            x_min = np.percentile(arr[:, 1], 5)
            x_max = np.percentile(arr[:, 1], 95)
            y_min = np.percentile(arr[:, 2], 10)
            y_max = np.percentile(arr[:, 2], 95)
        else:
            arr = np.array(data_list)
            x_min, x_max = 0, arr[:, 1].max()
            y_min, y_max = arr[:, 2].min(), arr[:, 2].max()
        
        x_span = x_max - x_min if x_max != x_min else 1.0
        y_span = y_max - y_min if y_max != y_min else 1.0

        ax.set_xlim(x_min + (x_span * 0.05), x_max + (x_span * 0.05))
        ax.set_ylim(y_min - (y_span * 0.1), y_max + (y_span * 0.1))

    @pyqtSlot(np.ndarray)
    def update_plot(self, eis_data):
        """Unpacks the 21-element array and updates all plots"""
        try:
            # Reconstruct Frequency from log10(Freq)
            Freq = 10 ** eis_data[1] 
            
            # --- Extract Data for 3 Cells ---
            # Indices match your server concatenation: [Row1(6), SoH1(1), Row2(6), SoH2(1), Row3(6), SoH3(1)]
            c1_real, c1_imag, c1_soh = eis_data[2], eis_data[3], eis_data[6]
            c2_real, c2_imag, c2_soh = eis_data[9], eis_data[10], eis_data[13]
            c3_real, c3_imag, c3_soh = eis_data[16], eis_data[17], eis_data[20]

            # Update SoH Labels
            self.style_soh_label(self.lbl_soh_c1, c1_soh, "Cell 1 SoH")
            self.style_soh_label(self.lbl_soh_c2, c2_soh, "Cell 2 SoH")
            self.style_soh_label(self.lbl_soh_c3, c3_soh, "Cell 3 SoH")

            # Update Progress (Using log math to prevent zero-span errors)
            start_log, stop_log = np.log10(10), np.log10(0.011) 
            fraction = (start_log - np.log10(Freq)) / (start_log - stop_log)
            self.progress_bar.setValue(max(0, min(100, int(fraction * 100))))

            # Save incoming point to respective lists
            self.data_c1.append((Freq, c1_real, c1_imag))
            self.data_c2.append((Freq, c2_real, c2_imag))
            self.data_c3.append((Freq, c3_real, c3_imag))

            # Update Scatter Plots & Zooms
            for data_list, scatter_obj, ax_obj, canvas_obj in [
                (self.data_c1, self.scatter1, self.ax1, self.canvas1),
                (self.data_c2, self.scatter2, self.ax2, self.canvas2),
                (self.data_c3, self.scatter3, self.ax3, self.canvas3)
            ]:
                if len(data_list) > 0:
                    arr = np.array(data_list)
                    # Set X = Z_real, Y = -Z_imag (Nyquist)
                    scatter_obj.set_offsets(np.column_stack((arr[:, 1], arr[:, 2])))
                    self.apply_smart_zoom(ax_obj, data_list)
                    canvas_obj.draw()

        except Exception as e:
            print(f"Update Plot Error: {e}")

    @pyqtSlot(str, str)
    def update_status_display(self, color, text):
        hex_c = {"red": "#FF5252", "orange": "#FF9800", "green": "#00E676"}.get(color, "grey")
        self.status_light.setStyleSheet(f"background-color: {hex_c}; border-radius: 10px;")
        self.status_text.setText(text)

    # --- BUTTON HANDLERS ---
    def send_start(self):
        # Clear previous data
        self.data_c1, self.data_c2, self.data_c3 = [], [], []
        
        for scatter_obj, canvas_obj in [(self.scatter1, self.canvas1), (self.scatter2, self.canvas2), (self.scatter3, self.canvas3)]:
            scatter_obj.set_offsets(np.empty((0, 2)))
            canvas_obj.draw()
        
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
        if not self.data_c1: return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")

        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Frequency (Hz)", 
                        "C1_Z_real", "C1_-Z_imag", 
                        "C2_Z_real", "C2_-Z_imag", 
                        "C3_Z_real", "C3_-Z_imag"
                    ])
                    # Iterate through the arrays simultaneously
                    for i in range(len(self.data_c1)):
                        writer.writerow([
                            self.data_c1[i][0], # Freq
                            self.data_c1[i][1], self.data_c1[i][2],
                            self.data_c2[i][1], self.data_c2[i][2],
                            self.data_c3[i][1], self.data_c3[i][2]
                        ])
                self.status_text.setText("Status: Data Saved!")
            except Exception as e:
                self.status_text.setText("Status: Error Saving")
                print(f"Error saving file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())