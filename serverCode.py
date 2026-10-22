from MyDigilent import FFT_peakAmp, MyDigilent, calculate_tolerated_impedance, dual_phase_demod, HolderCalibrator, smooth_impedance_array, find_converged_impedance, remove_baseline_full
from time import sleep
import numpy as np, socket, struct, mlrepo as ml

# SoH estimator
SoH_est = ml.NumpySimpleSoHLSTM('simple_soh_weights.npz')

# --- TCP Configuration ---
# 1. SERVER CONFIG: Listen on ALL interfaces
TCP_IP = '0.0.0.0'
TCP_PORT = 5005            

# --- Hardware Initialization ---
PIN_TX = 0            
PIN_RX = 1            
BAUDRATE = 115200

# Calibrator
_ref1 = np.array([
    [999.9998, 0.030041, -0.00309],
    [794.328, 0.030285, -0.00203],
    [630.9572, 0.030541, -0.0012],
    [501.1871, 0.030806, -0.00051],
    [398.107, 0.031075, 3.35e-05],
    [316.2276, 0.031378, 0.000458],
    [251.1885, 0.031666, 0.000817],
    [199.5261, 0.031965, 0.001084],
    [158.4892, 0.032261, 0.001318],
    [125.8925, 0.032557, 0.001504],
    [99.99994, 0.032863, 0.001703],
    [79.43277, 0.033181, 0.001852],
    [63.09569, 0.033493, 0.001986],
    [50.11869, 0.033829, 0.002111],
    [39.81069, 0.034154, 0.002244],
    [31.62275, 0.034521, 0.002362],
    [25.11884, 0.034882, 0.002467],
    [19.95261, 0.035268, 0.002578],
    [15.84892, 0.035669, 0.002681],
    [12.58924, 0.036088, 0.002793],
    [9.999991, 0.036578, 0.002899],
    [7.943275, 0.037048, 0.002921],
    [6.309568, 0.037513, 0.00301],
    [5.011868, 0.038053, 0.003037],
    [3.981068, 0.038604, 0.002996],
    [3.162275, 0.039096, 0.002916],
    [2.511884, 0.039582, 0.002797],
    [1.99526, 0.040025, 0.002636],
    [1.584892, 0.040373, 0.002457],
    [1.258924, 0.040728, 0.002337],
    [0.999999, 0.041016, 0.002167],
    [0.794327, 0.041285, 0.002072],
    [0.630957, 0.041516, 0.001961],
    [0.501187, 0.041748, 0.001929],
    [0.398107, 0.041911, 0.001884],
    [0.316227, 0.042136, 0.001918],
    [0.251188, 0.042296, 0.001988],
    [0.199526, 0.042501, 0.002054],
    [0.158489, 0.042737, 0.002223],
    [0.125892, 0.042925, 0.002441],
    [0.1, 0.043161, 0.002644],
    [0.079433, 0.043479, 0.002926],
    [0.063096, 0.043799, 0.003337],
    [0.050119, 0.044082, 0.003791],
    [0.039811, 0.044426, 0.004221],
    [0.031623, 0.044933, 0.004745],
    [0.025119, 0.045502, 0.005518],
    [0.019953, 0.045976, 0.006441],
    [0.015849, 0.046502, 0.007339],
    [0.012589, 0.047354, 0.008321],
    [0.01, 0.048505, 0.009681]
])
_ref2 = np.array([
    [999.9998, 0.029791, -0.00308],
    [794.328, 0.030014, -0.00204],
    [630.9572, 0.03027, -0.00122],
    [501.1871, 0.030533, -0.00056],
    [398.107, 0.030804, -1.50E-05],
    [316.2276, 0.031082, 0.000409],
    [251.1885, 0.031362, 0.000751],
    [199.5261, 0.031646, 0.00103],
    [158.4892, 0.031936, 0.001273],
    [125.8925, 0.032234, 0.001464],
    [99.99994, 0.032533, 0.001633],
    [79.43277, 0.032831, 0.001783],
    [63.09569, 0.033153, 0.001911],
    [50.11869, 0.03348, 0.00203],
    [39.81069, 0.033804, 0.002133],
    [31.62275, 0.034157, 0.002236],
    [25.11884, 0.034507, 0.002334],
    [19.95261, 0.034881, 0.002433],
    [15.84892, 0.035257, 0.002511],
    [12.58924, 0.035666, 0.002581],
    [9.999991, 0.036124, 0.002631],
    [7.943275, 0.03656, 0.002627],
    [6.309568, 0.036963, 0.002675],
    [5.011868, 0.03746, 0.002666],
    [3.981068, 0.037889, 0.002616],
    [3.162275, 0.038324, 0.002502],
    [2.511884, 0.038721, 0.00238],
    [1.99526, 0.039057, 0.002207],
    [1.584892, 0.03935, 0.00208],
    [1.258924, 0.03965, 0.002],
    [0.999999, 0.039878, 0.001859],
    [0.794327, 0.040117, 0.001806],
    [0.630957, 0.040299, 0.001718],
    [0.501187, 0.040504, 0.001717],
    [0.398107, 0.040657, 0.001714],
    [0.316227, 0.040859, 0.001772],
    [0.251188, 0.041008, 0.001868],
    [0.199526, 0.041205, 0.001956],
    [0.158489, 0.041442, 0.002142],
    [0.125892, 0.041626, 0.00237],
    [0.1, 0.041853, 0.002581],
    [0.079433, 0.042176, 0.002872],
    [0.063096, 0.042502, 0.003282],
    [0.050119, 0.042784, 0.003734],
    [0.039811, 0.043133, 0.004161],
    [0.031623, 0.043636, 0.004685],
    [0.025119, 0.044197, 0.005437],
    [0.019953, 0.044671, 0.006349],
    [0.015849, 0.045189, 0.007226],
    [0.012589, 0.046022, 0.008201],
    [0.01, 0.047152, 0.009549]
])
_ref3 = np.array([
    [999.9989, 0.065103, -0.00387],
    [794.3279, 0.065368, -0.00261],
    [630.9565, 0.065635, -0.0016],
    [501.1869, 0.065934, -0.0008],
    [398.1069, 0.066235, -0.00015],
    [316.2277, 0.066558, 0.000368],
    [251.1886, 0.066882, 0.000774],
    [199.5264, 0.067231, 0.001129],
    [158.4893, 0.067553, 0.001418],
    [125.8926, 0.067879, 0.001666],
    [100, 0.068196, 0.001834],
    [79.43272, 0.068578, 0.002005],
    [63.09565, 0.068942, 0.00215],
    [50.11861, 0.069291, 0.002296],
    [39.81068, 0.069696, 0.002408],
    [31.62277, 0.070088, 0.002511],
    [25.11888, 0.070445, 0.002622],
    [19.95259, 0.070898, 0.002722],
    [15.8489, 0.071313, 0.002812],
    [12.58923, 0.07171, 0.002863],
    [9.999985, 0.072147, 0.002988],
    [7.943276, 0.072727, 0.00302],
    [6.309574, 0.073244, 0.003043],
    [5.011873, 0.073755, 0.003045],
    [3.981063, 0.074254, 0.002999],
    [3.162279, 0.074751, 0.002894],
    [2.511882, 0.075209, 0.002779],
    [1.995261, 0.075664, 0.002631],
    [1.584891, 0.076051, 0.00245],
    [1.258925, 0.076393, 0.002303],
    [0.999998, 0.076658, 0.002129],
    [0.794327, 0.07694, 0.002013],
    [0.630956, 0.077126, 0.001917],
    [0.501187, 0.077355, 0.001865],
    [0.398107, 0.077497, 0.00181],
    [0.316228, 0.077726, 0.001876],
    [0.251188, 0.077863, 0.001911],
    [0.199526, 0.078079, 0.002056],
    [0.158489, 0.078249, 0.002177],
    [0.125893, 0.07849, 0.002438],
    [0.1, 0.078701, 0.002638],
    [0.079433, 0.079, 0.003006],
    [0.063096, 0.079273, 0.003315],
    [0.050119, 0.079633, 0.003818],
    [0.039811, 0.079951, 0.004258],
    [0.031623, 0.080452, 0.004979],
    [0.025119, 0.080869, 0.005596],
    [0.019953, 0.081488, 0.006612],
    [0.015849, 0.082161, 0.007478],
    [0.012589, 0.083024, 0.00891],
    [0.01, 0.083983, 0.010088]
])

# Format: [Freq, Zreal, Zimag]
_meas_holder1 = np.array([
    [999.9866, 0.089341, -2.42e-05],
    [794.3237, 0.094177, -7.54e-06],
    [630.9536, 0.092053, -2.15e-06],
    [501.1821, 0.090873, -2.67e-05],
    [398.1049, 0.091648, 1.09e-05],
    [316.2241, 0.092666, -1.51e-05],
    [251.1862, 0.093376, -1.46e-06],
    [199.5249, 0.092424, 2.11e-05],
    [158.4888, 0.093781, 3.37e-05],
    [125.8923, 0.095298, 5.24e-05],
    [100.0, 0.095377, 4.25e-05],
    [79.43237, 0.09703, 5.70e-05],
    [63.0957, 0.096305, 5.35e-05],
    [50.11826, 0.097418, 4.57e-05],
    [39.81018, 0.098139, 6.27e-05],
    [31.62256, 0.097632, 5.18e-05],
    [25.11847, 0.098949, 4.56e-05],
    [19.95239, 0.099531, 6.21e-05],
    [15.84888, 0.099568, 7.20e-05],
    [12.58923, 0.100908, 4.99e-05],
    [10.0, 0.103536, 6.72e-05],
    [7.943237, 0.10266, 8.19e-05],
    [6.30954, 0.101882, 6.09e-05],
    [5.011871, 0.103139, 6.27e-05],
    [3.981018, 0.103307, 7.53e-05],
    [3.162231, 0.105275, 5.56e-05],
    [2.511871, 0.103119, 0.000108],
    [1.995239, 0.113185, 5.94e-05],
    [1.584869, 0.105069, 1.06e-05],
    [1.258911, 0.115724, 0.0002],
    [1.0, 0.107615, 4.90e-05],
    [0.794312, 0.115408, 0.000124],
    [0.630951, 0.109134, 0.000143],
    [0.50116, 0.113315, 0.000118],
    [0.398102, 0.108254, 1.84e-05],
    [0.316223, 0.108791, 0.000107],
    [0.25116, 0.104743, 8.45e-05],
    [0.199524, 0.106663, 4.94e-05],
    [0.158478, 0.11736, 7.92e-05],
    [0.125885, 0.108923, 5.33e-05],
    [0.099976, 0.111075, -1.39e-05],
    [0.079407, 0.11402, 4.35e-05],
    [0.06308, 0.107643, 6.77e-05],
    [0.05011, 0.112372, 7.35e-05],
    [0.039795, 0.111441, 3.69e-05],
    [0.031616, 0.110564, 0.000164],
    [0.025116, 0.117951, 7.52e-05],
    [0.019928, 0.111068, 0.000133],
    [0.015839, 0.112634, 0.000191],
    [0.012573, 0.114163, 0.00018],
    [0.009979, 0.118836, 0.000191]
])
_meas_holder2 = np.array([
    [999.9866, 0.085132, -4.50e-05],
    [794.3237, 0.088932, -2.51e-05],
    [630.9536, 0.087033, -9.54e-06],
    [501.1821, 0.086191, -3.34e-05],
    [398.1049, 0.08658, 5.17e-06],
    [316.2241, 0.08775, -1.64e-05],
    [251.1862, 0.088356, -3.25e-06],
    [199.5249, 0.087234, 1.37e-05],
    [158.4888, 0.088951, 3.24e-05],
    [125.8923, 0.090297, 5.05e-05],
    [100.0, 0.0907, 4.32e-05],
    [79.43237, 0.092039, 5.57e-05],
    [63.0957, 0.090801, 5.31e-05],
    [50.11826, 0.092313, 4.62e-05],
    [39.81018, 0.092629, 5.94e-05],
    [31.62256, 0.092491, 5.02e-05],
    [25.11847, 0.093757, 4.92e-05],
    [19.95239, 0.094276, 6.23e-05],
    [15.84888, 0.094648, 7.88e-05],
    [12.58923, 0.096042, 5.06e-05],
    [10.0, 0.098548, 7.14e-05],
    [7.943237, 0.097305, 8.86e-05],
    [6.30954, 0.09636, 7.19e-05],
    [5.011871, 0.097678, 5.01e-05],
    [3.981018, 0.098113, 5.87e-05],
    [3.162231, 0.100381, 6.34e-05],
    [2.511871, 0.097957, 0.000121],
    [1.995239, 0.108698, 9.02e-05],
    [1.584869, 0.100134, 9.77e-06],
    [1.258911, 0.109852, 0.000188],
    [1.0, 0.102225, 5.33e-05],
    [0.794312, 0.111571, 0.00011],
    [0.630951, 0.103661, 0.000154],
    [0.50116, 0.107723, 9.89e-05],
    [0.398102, 0.103474, 1.76e-05],
    [0.316223, 0.101871, 0.000115],
    [0.25116, 0.09966, 7.08e-05],
    [0.199524, 0.102752, 4.38e-05],
    [0.158478, 0.111727, 8.56e-05],
    [0.125885, 0.104877, 5.78e-05],
    [0.099976, 0.10362, -3.33e-06],
    [0.079407, 0.106249, 4.78e-05],
    [0.06308, 0.102779, 4.33e-05],
    [0.05011, 0.1076, 7.02e-05],
    [0.039795, 0.106338, 2.16e-05],
    [0.031616, 0.105818, 0.000149],
    [0.025116, 0.110901, 6.91e-05],
    [0.019928, 0.105874, 0.000132],
    [0.015839, 0.107518, 0.000173],
    [0.012573, 0.110103, 0.000217],
    [0.009979, 0.113131, 0.000191]
])
_meas_holder3 = np.array([
    [999.9866, 0.1058, -4.59e-05],
    [794.3237, 0.111097, -2.13e-05],
    [630.9536, 0.108354, -1.23e-05],
    [501.1821, 0.108518, -3.55e-05],
    [398.1049, 0.109031, 6.35e-06],
    [316.2241, 0.110175, -1.89e-05],
    [251.1862, 0.109998, -2.07e-06],
    [199.5249, 0.10903, 1.88e-05],
    [158.4888, 0.111027, 4.31e-05],
    [125.8923, 0.111793, 6.36e-05],
    [100.0, 0.11187, 5.74e-05],
    [79.43237, 0.113873, 6.88e-05],
    [63.0957, 0.11242, 6.26e-05],
    [50.11826, 0.113728, 5.63e-05],
    [39.81018, 0.114683, 7.93e-05],
    [31.62256, 0.113718, 6.61e-05],
    [25.11847, 0.115817, 6.11e-05],
    [19.95239, 0.116133, 7.30e-05],
    [15.84888, 0.116555, 9.02e-05],
    [12.58923, 0.117654, 6.37e-05],
    [10.0, 0.120666, 9.05e-05],
    [7.943237, 0.119967, 9.62e-05],
    [6.30954, 0.11895, 8.27e-05],
    [5.011871, 0.120615, 6.99e-05],
    [3.981018, 0.120724, 7.76e-05],
    [3.162231, 0.123301, 8.88e-05],
    [2.511871, 0.119392, 0.000146],
    [1.995239, 0.133813, 0.000114],
    [1.584869, 0.122774, 3.30e-05],
    [1.258911, 0.135859, 0.000249],
    [1.0, 0.128309, 8.31e-05],
    [0.794312, 0.136386, 0.000155],
    [0.630951, 0.128439, 0.00018],
    [0.50116, 0.134453, 0.000157],
    [0.398102, 0.130024, 3.79e-05],
    [0.316223, 0.127997, 0.000144],
    [0.25116, 0.124312, 0.000108],
    [0.199524, 0.128449, 5.86e-05],
    [0.158478, 0.140031, 0.000125],
    [0.125885, 0.130132, 7.12e-05],
    [0.099976, 0.130798, -1.35e-05],
    [0.079407, 0.133045, 7.71e-05],
    [0.06308, 0.128636, 8.52e-05],
    [0.05011, 0.135397, 0.000122],
    [0.039795, 0.133904, 7.66e-05],
    [0.031616, 0.133764, 0.000221],
    [0.025116, 0.140135, 0.000132],
    [0.019928, 0.134235, 0.000174],
    [0.015839, 0.134694, 0.000247],
    [0.012573, 0.13844, 0.000284],
    [0.009979, 0.145292, 0.000292]
])

# Initialize the three calibrators ONCE during startup
calibrator_c1 = HolderCalibrator(_ref1, smooth_impedance_array(_meas_holder1, window_size=5), name="Cell 1")
calibrator_c2 = HolderCalibrator(_ref2, smooth_impedance_array(_meas_holder2, window_size=5), name="Cell 2")
calibrator_c3 = HolderCalibrator(_ref3, smooth_impedance_array(_meas_holder3, window_size=5), name="Cell 3")

# Initialize Hardware ONCE (before entering the network loop)
print("Initializing Digilent-ADP3450...")
Digi_1 = MyDigilent(tx=PIN_TX, rx=PIN_RX, baud_rate=BAUDRATE, parity="none", data_bits=8, stop_bits=1)
Digi_1.scope_setup(channels=[1, 2, 3, 4])
sleep(1)

max_buf = Digi_1.dev.analog.input.max_buffer_size
fsample_max = 1e6
print(f"Max buffer size per channel: {max_buf}, Max sampling rate: {fsample_max}")

# --- Frequency Setup ---
f_freq = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
# f_freq = [1e3]
finit_idx = 0
fperdecade = 10
FREQ_TEMPLATE = []
FREQ_TEMPLATE.append(f_freq[finit_idx])
for i in range(finit_idx, len(f_freq)-1):
    for k in range(1, fperdecade+1):
        FREQ_TEMPLATE.append(10**(np.log10(f_freq[i]).item()-k/fperdecade))

# Track the previous impedance for each channel independently
Z1 = None
Z2 = None
Z3 = None

# --- MAIN SERVER LOOP ---
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server_sock.bind((TCP_IP, TCP_PORT))
    server_sock.listen(1)
    print(f"Server listening on {TCP_IP}:{TCP_PORT}")
    print("Digilent is ready. Waiting for Client connection...")

    # 2. OUTER LOOP: Keeps the server alive forever
    while True:
        try:
            # Block here until a Client connects
            conn, addr = server_sock.accept()
            print(f"\nClient connected from: {addr}")
            
            client_connected = True

            # 3. CLIENT SESSION LOOP
            while client_connected:
                print("--- STANDBY: Waiting for 'START' command ---")
                
                # Ensure blocking mode while waiting for command
                conn.setblocking(True) 
                
                try:
                    command_raw = conn.recv(1024)
                    if not command_raw:
                        print("Client disconnected.")
                        client_connected = False
                        break 
                    
                    command = command_raw.decode('utf-8').strip()
                except ConnectionResetError:
                    print("Connection reset by peer.")
                    client_connected = False
                    break
                except Exception as e:
                    print(f"Receive Error: {e}")
                    client_connected = False
                    break

                if command == "START":
                    print("START received. Beginning Measurement Sequence...")
                    
                    # Initialize run variables
                    sample_c1 = np.zeros([51, 6])
                    sample_c2 = np.zeros([51, 6])
                    sample_c3 = np.zeros([51, 6])
                    i_idx = 0
                    stop_requested = False

                    # Loop through frequencies
                    for f in FREQ_TEMPLATE:
                        # --- CHECK FOR STOP COMMAND (Non-Blocking) ---
                        try:
                            conn.setblocking(False) # Peek using the CLIENT connection
                            cmd_check = conn.recv(1024) 
                            if cmd_check and "STOP" in cmd_check.decode('utf-8'):
                                print("\n!!! STOP command received. Halting measurement !!!")
                                stop_requested = True
                                conn.setblocking(True)
                                break 
                        except BlockingIOError:
                            pass 
                        except Exception as e:
                            print(f"Socket error during check (Client likely disconnected): {e}")
                            stop_requested = True
                            client_connected = False
                            break
                        
                        conn.setblocking(True) # Restore blocking
                        # ---------------------------------------------

                        # Hardware Logic (Same as before)
                        CMD = str(f)
                        Digi_1.sendStringUART(CMD)
                        sleep(1) 
                        
                        mainloop = True
                        while mainloop:
                            RES = bytes(Digi_1.uart_read())
                            res_str = RES.decode("utf-8")

                            if res_str == "Received":
                                print(f"Measuring EIS at {CMD.strip()} Hz...")
                                buffer_size = int(max_buf)
                                if f > 6500:
                                    # Handles the sudden drop at the top of the table (e.g., 10000Hz -> 3023)
                                    ncycle = int(0.3023 * f)
                                    sample_rate = int(f*buffer_size/ncycle)
                                elif f >= 2.0:
                                    # The main linear regime (e.g., 100Hz -> 50, 1000Hz -> 503)
                                    ncycle = int(np.round(0.504 * f))
                                else:
                                    # Hard cap for very low frequencies
                                    ncycle = 1
                                sample_rate = int(f*buffer_size/ncycle)
                                sleep(0.5)

                                data_sets = Digi_1.scope_record(sample_rate, buffer_size)
                                print(f"buffer size: {buffer_size}, Perturbation freq: {f}, Sampling frequency: {sample_rate}, Number of cycles: {ncycle}")
                                Digi_1.sendStringUART("STOP")
                            
                            elif res_str == "DoneRecv":
                                # extract csv
                                # csv_data = np.column_stack(data_sets)

                                # np.savetxt("results/raw_data.csv", 
                                #             csv_data, 
                                #             delimiter=",", 
                                #             header="Current (A),Voltage_B1(V),Voltage_B2(V),Voltage_B3(V)", 
                                #             comments="", 
                                #             fmt="%.6f")

                                # Calculation Logic
                                Imeas = (data_sets[0] - np.mean(data_sets[0])) / -0.033
                                V1meas = data_sets[1] - np.mean(data_sets[1])
                                V2meas = data_sets[2] - np.mean(data_sets[2])
                                V3meas = data_sets[3] - np.mean(data_sets[3])

                                # Imeas, _ = remove_baseline_full(Imeas, sample_rate, f)
                                # V1meas, _ = remove_baseline_full(V1meas, sample_rate, f)
                                # V2meas, _ = remove_baseline_full(V2meas, sample_rate, f)
                                # V3meas, _ = remove_baseline_full(V3meas, sample_rate, f)

                                fi, I_amp, I_phase, I_comp = FFT_peakAmp(
                                                                buffer=Imeas,
                                                                commanded_freq=f,
                                                                sample_rate=sample_rate,
                                                                band_factor=0.05  # Searches between 900 Hz and 1100 Hz
                                                            )
                                
                                f1, V1_amp, V1_phase, V1_comp = FFT_peakAmp(
                                                                buffer=V1meas,
                                                                commanded_freq=f,
                                                                sample_rate=sample_rate,
                                                                band_factor=0.05  # Searches between 900 Hz and 1100 Hz
                                                            )
                                f2, V2_amp, V2_phase, V2_comp = FFT_peakAmp(
                                                                buffer=V2meas,
                                                                commanded_freq=f,
                                                                sample_rate=sample_rate,
                                                                band_factor=0.05  # Searches between 900 Hz and 1100 Hz
                                                            )
                                f3, V3_amp, V3_phase, V3_comp = FFT_peakAmp(
                                                                buffer=V3meas,
                                                                commanded_freq=f,
                                                                sample_rate=sample_rate,
                                                                band_factor=0.05  # Searches between 900 Hz and 1100 Hz
                                                            )
                                
                                # Imeas_filtered = I_amp * np.cos(2.0 * np.pi * f1 * np.arange(buffer_size)/sample_rate + np.radians(I_phase))
                                # V1meas_filtered = V1_amp * np.cos(2.0 * np.pi * f1 * np.arange(buffer_size)/sample_rate + np.radians(V1_phase))
                                # V2meas_filtered = V2_amp * np.cos(2.0 * np.pi * f2 * np.arange(buffer_size)/sample_rate + np.radians(V2_phase))
                                # V3meas_filtered = V3_amp * np.cos(2.0 * np.pi * f3 * np.arange(buffer_size)/sample_rate + np.radians(V3_phase))

                                # Imeas_filtered = Imeas
                                # V1meas_filtered = V1meas
                                # V2meas_filtered = V2meas
                                # V3meas_filtered = V3meas

                                # extract csv
                                # csv_data = np.column_stack([Imeas_filtered, V1meas_filtered, V2meas_filtered, V3meas_filtered])

                                # np.savetxt("results/filtered_data.csv", 
                                #             csv_data,
                                #             delimiter=",", 
                                #             header="Current (A),Voltage_B1(V),Voltage_B2(V),Voltage_B3(V)", 
                                #             comments="", 
                                #             fmt="%.6f")
                                
                                # rng_int = 0.005
                                
                                # # Process Voltage 1
                                # f1, Z1_complex, (z1_real, z1_imag), conv1 = find_converged_impedance(
                                #     v_buffer=V1meas, i_buffer=Imeas, commanded_freq=f, tolerance=0.01,
                                #     sample_rate=sample_rate, prev_impedance=prev_Z1, search_range=rng_int
                                # )
                                
                                # # Process Voltage 2
                                # f2, Z2_complex, (z2_real, z2_imag), conv2 = find_converged_impedance(
                                #     v_buffer=V2meas, i_buffer=Imeas, commanded_freq=f, tolerance=0.01,
                                #     sample_rate=sample_rate, prev_impedance=prev_Z2, search_range=rng_int
                                # )
                                
                                # # Process Voltage 3
                                # f3, Z3_complex, (z3_real, z3_imag), conv3 = find_converged_impedance(
                                #     v_buffer=V3meas, i_buffer=Imeas, commanded_freq=f, tolerance=0.01,
                                #     sample_rate=sample_rate, prev_impedance=prev_Z3, search_range=rng_int
                                # )
                                
                                # # Update previous values for the next frequency step
                                # prev_Z1 = Z1_complex
                                # prev_Z2 = Z2_complex
                                # prev_Z3 = Z3_complex
                                
                                # Warn if any channel failed to converge
                                # if not (conv1 and conv2 and conv3):
                                #     print(f"Warning: Convergence failed at {f}Hz. Using best available fits.")

                                # buffer_size = Imeas.shape[0]

                                # sfreq = (f1+f2+f3) / 3.0  # Average frequency from the three channels
                                
                                I_amp, I_phase = dual_phase_demod(Imeas, fi, sample_rate)
                                V1_amp, V1_phase = dual_phase_demod(V1meas, f1, sample_rate)
                                V2_amp, V2_phase = dual_phase_demod(V2meas, f2, sample_rate)
                                V3_amp, V3_phase = dual_phase_demod(V3meas, f3, sample_rate)

                                print(f"Freq: [{f1:.5f}, {f2:.5f}, {f3:.5f}] Hz | V_amp: [{V1_amp:.2E}, {V2_amp:.2E}, {V3_amp:.2E}] | I_amp: {I_amp:.2E}")

                                # # if np.max(V1meas) > 1:
                                # #     break

                                I_real = I_amp * np.cos(np.radians(I_phase))
                                I_imag = I_amp * np.sin(np.radians(I_phase))
                                V1_real = V1_amp * np.cos(np.radians(V1_phase))
                                V1_imag = V1_amp * np.sin(np.radians(V1_phase))
                                V2_real = V2_amp * np.cos(np.radians(V2_phase))
                                V2_imag = V2_amp * np.sin(np.radians(V2_phase))
                                V3_real = V3_amp * np.cos(np.radians(V3_phase))
                                V3_imag = V3_amp * np.sin(np.radians(V3_phase))

                                V1_comp = V1_real + 1j * V1_imag
                                V2_comp = V2_real + 1j * V2_imag
                                V3_comp = V3_real + 1j * V3_imag
                                I_comp = I_real + 1j * I_imag
                                
                                Z1 = V1_comp / I_comp
                                Z2 = V2_comp / I_comp
                                Z3 = V3_comp / I_comp

                                # Z1real, Z1imag = calibrator_c1.correct(f1, Z1.real, -Z1.imag)
                                # print(f"Cell-1 Impedance: {Z1real} + ({Z1imag}j)")

                                # Z2real, Z2imag = calibrator_c2.correct(f2, Z2.real, -Z2.imag)
                                # print(f"Cell-2 Impedance: {Z2real} + ({Z2imag}j)")

                                # Z3real, Z3imag = calibrator_c3.correct(f3, Z3.real, -Z3.imag)
                                # print(f"Cell-3 Impedance: {Z3real} + ({Z3imag}j)")
                                
                                # Pass the previous Z and the tolerance into the function
                                # Z1 = calculate_tolerated_impedance(
                                #     v_buffer=V1meas, 
                                #     i_buffer=Imeas, 
                                #     target_freq=f,
                                #     sample_rate=sample_rate, 
                                #     prev_Z_complex=Z1, 
                                #     tolerance=0.03  # 3% boundary
                                # )
                                
                                # Z2 = calculate_tolerated_impedance(
                                #     v_buffer=V2meas, 
                                #     i_buffer=Imeas, 
                                #     target_freq=f,
                                #     sample_rate=sample_rate, 
                                #     prev_Z_complex=Z2, 
                                #     tolerance=0.03  # 3% boundary
                                # )
                                
                                # Z3 = calculate_tolerated_impedance(
                                #     v_buffer=V3meas, 
                                #     i_buffer=Imeas, 
                                #     target_freq=f,
                                #     sample_rate=sample_rate, 
                                #     prev_Z_complex=Z3, 
                                #     tolerance=0.03  # 3% boundary
                                # )
                                
                                Z1real, Z1imag = Z1.real, -Z1.imag
                                Z2real, Z2imag = Z2.real, -Z2.imag
                                Z3real, Z3imag = Z3.real, -Z3.imag

                                # Z1real, Z1imag = z1_real, z1_imag
                                # Z2real, Z2imag = z2_real, z2_imag
                                # Z3real, Z3imag = z3_real, z3_imag
                                
                                

                                # Data Quality Check
                                if i_idx > 0 and ((Z1real < 0.9*sample_c1[i_idx-1, 1] and Z1real < 0) or (Z2real < 0.9*sample_c2[i_idx-1, 1] and Z2real < 0) or (Z3real < 0.9*sample_c3[i_idx-1, 1] and Z3real < 0)):
                                    print("\nFrequency skipped (Impedance Drop)...\n")
                                    break
                                
                                sample_c1[i_idx, 0] = np.mean(data_sets[1])
                                sample_c1[i_idx, 1] = np.log10(f1)
                                sample_c1[i_idx, 2] = Z1real
                                sample_c1[i_idx, 3] = Z1imag
                                sample_c1[i_idx, 4] = np.abs(Z1real - 1j*Z1imag)
                                sample_c1[i_idx, 5] = np.angle(Z1real - 1j*Z1imag, deg=True)

                                sample_c2[i_idx, 0] = np.mean(data_sets[2])
                                sample_c2[i_idx, 1] = np.log10(f2)
                                sample_c2[i_idx, 2] = Z2real
                                sample_c2[i_idx, 3] = Z2imag
                                sample_c2[i_idx, 4] = np.abs(Z2real - 1j*Z2imag)
                                sample_c2[i_idx, 5] = np.angle(Z2real - 1j*Z2imag, deg=True)

                                sample_c3[i_idx, 0] = np.mean(data_sets[3])
                                sample_c3[i_idx, 1] = np.log10(f3)
                                sample_c3[i_idx, 2] = Z3real
                                sample_c3[i_idx, 3] = Z3imag
                                sample_c3[i_idx, 4] = np.abs(Z3real - 1j*Z3imag)
                                sample_c3[i_idx, 5] = np.angle(Z3real - 1j*Z3imag, deg=True)

                                # --- ML based SoH estimation ---
                                f_idx = [0, 1, 4, 5]
                                
                                # --- ML based SoH estimation ---
                                input_c1 = sample_c1[:, f_idx].T.reshape(1, 4, 51).astype(np.float32)
                                output_c1 = SoH_est.predict(input_c1)
                                print(f"\n\nThe estimated SoH of cell-1 is: {str(np.round(output_c1*100, decimals=2))}%\n")

                                input_c2 = sample_c2[:, f_idx].T.reshape(1, 4, 51).astype(np.float32)
                                output_c2 = SoH_est.predict(input_c2)
                                print(f"\n\nThe estimated SoH of cell-2 is: {str(np.round(output_c2*100, decimals=2))}%\n")

                                input_c3 = sample_c3[:, f_idx].T.reshape(1, 4, 51).astype(np.float32)
                                output_c3 = SoH_est.predict(input_c3)
                                print(f"\n\nThe estimated SoH of cell-3 is: {str(np.round(output_c3*100, decimals=2))}%\n")
                                
                                # --- Send Data to Host ---
                                try:
                                    # 1. Extract the current row for each cell (6 values each)
                                    row_c1 = sample_c1[i_idx, :].flatten()
                                    row_c2 = sample_c2[i_idx, :].flatten()
                                    row_c3 = sample_c3[i_idx, :].flatten()
                                    
                                    # 2. Format the SoH estimates (clip, scale to %, round, and flatten)
                                    soh_1 = np.round(np.clip(output_c1, 0, 1) * 100, decimals=2).flatten()
                                    soh_2 = np.round(np.clip(output_c2, 0, 1) * 100, decimals=2).flatten()
                                    soh_3 = np.round(np.clip(output_c3, 0, 1) * 100, decimals=2).flatten()
                                    
                                    # 3. Concatenate everything into one flat array
                                    # Total length will be (6 + 1) * 3 = 21 elements
                                    combined_data = np.concatenate([
                                        row_c1, soh_1, 
                                        row_c2, soh_2, 
                                        row_c3, soh_3
                                    ])
                                    
                                    # 4. Convert to bytes (ensure consistent float64 type for struct unpacking)
                                    data_bytes = combined_data.astype(np.float64).tobytes()
                                    header = struct.pack('>I', len(data_bytes))
                                    
                                    # 5. Send over TCP
                                    conn.sendall(header + data_bytes)
                                    print(f"Sent combined measurements (21 values) to Client.")
                                    
                                except Exception as e:
                                    print(f"Send failed (Client disconnected?): {e}")
                                    client_connected = False
                                    stop_requested = True # Force loop exit

                                i_idx += 1
                                mainloop = False 
                                sleep(3)

                        if not client_connected: break

                    print("Sequence finished or stopped.")
                    # Data saving logic (optional)...

            # End of Client Session
            if conn: conn.close()
            print("Session ended. Returning to wait state...\n")

        except Exception as e:
            print(f"Server Loop Error: {e}")
            # Ensure we don't crash the server script
            try: conn.close()
            except: pass

except KeyboardInterrupt:
    print("\nServer shutting down manually.")
finally:
    Digi_1.close()
    server_sock.close()
    print("Device and Socket closed.")