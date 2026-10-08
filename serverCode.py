from MyDigilent import MyDigilent, remove_baseline_full, freq_selection_signal, dual_phase_demod, FFT, fir_bandpass, HolderCalibrator, smooth_impedance_array
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
    [852.0624, 0.081472, -0.00358],
    [769.1927, 0.088676, -0.00239],
    [769.1927, 0.086348, -0.00186],
    [769.1927, 0.086795, -0.0035],
    [769.1927, 0.08693, -0.00266],
    [313.6102, 0.085705, -0.00205],
    [251.6289, 0.087821, 0.000596],
    [199.2901, 0.088227, 0.001208],
    [159.0294, 0.088261, 0.001862],
    [126.821, 0.088224, 0.001998],
    [100, 0.089134, 0.0023],
    [79.43262, 0.08988, 0.002682],
    [63.09558, 0.090248, 0.002638],
    [50.11826, 0.090803, 0.002722],
    [39.80991, 0.091264, 0.002831],
    [31.6221, 0.091802, 0.002737],
    [25.11841, 0.092211, 0.003156],
    [19.95209, 0.092508, 0.003049],
    [15.84888, 0.092929, 0.003158],
    [12.58881, 0.093788, 0.003267],
    [9.999298, 0.09411, 0.003658],
    [7.942383, 0.095435, 0.003517],
    [6.308899, 0.095248, 0.003714],
    [5.011597, 0.095964, 0.003931],
    [3.980621, 0.096611, 0.003552],
    [3.161621, 0.097126, 0.003414],
    [2.511414, 0.097647, 0.003652],
    [1.99469, 0.098778, 0.003106],
    [1.584473, 0.098753, 0.00319],
    [1.25882, 0.099009, 0.002861],
    [0.999725, 0.099618, 0.002689],
    [0.79422, 0.100055, 0.002393],
    [0.630615, 0.100504, 0.002438],
    [0.500977, 0.100546, 0.002387],
    [0.397949, 0.100225, 0.003041],
    [0.316132, 0.101401, 0.001694],
    [0.251007, 0.100891, 0.002453],
    [0.199402, 0.101747, 0.002497],
    [0.158447, 0.101893, 0.002666],
    [0.125854, 0.100489, 0.003672],
    [0.100199, 0.102918, 0.001428],
    [0.079591, 0.105108, 0.00661],
    [0.063221, 0.102415, 0.004248],
    [0.050219, 0.103311, 0.003306],
    [0.03989, 0.10255, 0.003562],
    [0.031686, 0.100855, 0.007602],
    [0.025169, 0.105852, 0.007459],
    [0.019992, 0.104486, 0.007466],
    [0.015881, 0.105918, 0.008388],
    [0.012614, 0.110244, 0.005466],
    [0.01002, 0.106897, 0.010569]
])
_meas_holder2 = np.array([
    [852.0624, 0.084388, -0.00533],
    [769.1927, 0.091371, -0.0037],
    [769.1927, 0.088438, -0.0031],
    [769.1927, 0.088953, -0.00479],
    [769.1927, 0.088775, -0.004],
    [313.6102, 0.08778, -0.00375],
    [251.6289, 0.089986, 6.70e-05],
    [199.2901, 0.090164, 0.000795],
    [159.0294, 0.090244, 0.001608],
    [126.821, 0.089934, 0.001729],
    [100, 0.090386, 0.002119],
    [79.43262, 0.090975, 0.002539],
    [63.09558, 0.091256, 0.002643],
    [50.11826, 0.091745, 0.002629],
    [39.80991, 0.092119, 0.00278],
    [31.6221, 0.092583, 0.002711],
    [25.11841, 0.093088, 0.003082],
    [19.95209, 0.093251, 0.002976],
    [15.84888, 0.09366, 0.003001],
    [12.58881, 0.094303, 0.002971],
    [9.999298, 0.094714, 0.003383],
    [7.942383, 0.095985, 0.003223],
    [6.308899, 0.095714, 0.00342],
    [5.011597, 0.096197, 0.003586],
    [3.980621, 0.096868, 0.003145],
    [3.161621, 0.09709, 0.002988],
    [2.511414, 0.09742, 0.00318],
    [1.99469, 0.098724, 0.002651],
    [1.584473, 0.098609, 0.002815],
    [1.25882, 0.09869, 0.00254],
    [0.999725, 0.099003, 0.002319],
    [0.79422, 0.099365, 0.001857],
    [0.630615, 0.099603, 0.002222],
    [0.500977, 0.099642, 0.002208],
    [0.397949, 0.099417, 0.002429],
    [0.316132, 0.100363, 0.001295],
    [0.251007, 0.100003, 0.002418],
    [0.199402, 0.101034, 0.002294],
    [0.158447, 0.101474, 0.003179],
    [0.125854, 0.099859, 0.002786],
    [0.100199, 0.102139, 0.00163],
    [0.079591, 0.105207, 0.005634],
    [0.063221, 0.103765, 0.004571],
    [0.050219, 0.102971, 0.002716],
    [0.03989, 0.101561, 0.004709],
    [0.031686, 0.10143, 0.007484],
    [0.025169, 0.104854, 0.005905],
    [0.019992, 0.103346, 0.007523],
    [0.015881, 0.104847, 0.009101],
    [0.012614, 0.109445, 0.007157],
    [0.01002, 0.104642, 0.011275]
])
_meas_holder3 = np.array([
    [852.0624, 0.106066, -0.00551],
    [769.1927, 0.102543, -0.00349],
    [769.1927, 0.099881, -0.0023],
    [769.1927, 0.099616, -0.00447],
    [769.1927, 0.099956, -0.00345],
    [313.6102, 0.104017, -0.0038],
    [251.6289, 0.107932, 0.000374],
    [199.2901, 0.109296, 0.001105],
    [159.0294, 0.109734, 0.002081],
    [126.821, 0.110053, 0.002256],
    [100, 0.111168, 0.002642],
    [79.43262, 0.112201, 0.003146],
    [63.09558, 0.112606, 0.003222],
    [50.11826, 0.113269, 0.00318],
    [39.80991, 0.113759, 0.003337],
    [31.6221, 0.114313, 0.0031],
    [25.11841, 0.114678, 0.003595],
    [19.95209, 0.114992, 0.003402],
    [15.84888, 0.115312, 0.003511],
    [12.58881, 0.116086, 0.003595],
    [9.999298, 0.116496, 0.004022],
    [7.942383, 0.117873, 0.003877],
    [6.308899, 0.117528, 0.004218],
    [5.011597, 0.118181, 0.004383],
    [3.980621, 0.118956, 0.00392],
    [3.161621, 0.119235, 0.003866],
    [2.511414, 0.119708, 0.004169],
    [1.99469, 0.121058, 0.003536],
    [1.584473, 0.121145, 0.003702],
    [1.25882, 0.12113, 0.003201],
    [0.999725, 0.12153, 0.003077],
    [0.79422, 0.122011, 0.002472],
    [0.630615, 0.122493, 0.002757],
    [0.500977, 0.122043, 0.002452],
    [0.397949, 0.121505, 0.003049],
    [0.316132, 0.122792, 0.001413],
    [0.251007, 0.122181, 0.002458],
    [0.199402, 0.122968, 0.002322],
    [0.158447, 0.123213, 0.002945],
    [0.125854, 0.121394, 0.002824],
    [0.100199, 0.123293, 0.00076],
    [0.079591, 0.12685, 0.006096],
    [0.063221, 0.123822, 0.005183],
    [0.050219, 0.123055, 0.0031],
    [0.03989, 0.12285, 0.00348],
    [0.031686, 0.12074, 0.007113],
    [0.025169, 0.126231, 0.005791],
    [0.019992, 0.124577, 0.006948],
    [0.015881, 0.125346, 0.009183],
    [0.012614, 0.130207, 0.0065],
    [0.01002, 0.126667, 0.01026]
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
# f_freq = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
f_freq = [1e3]
finit_idx = 0
fperdecade = 10
FREQ_TEMPLATE = []
FREQ_TEMPLATE.append(f_freq[finit_idx])
for i in range(finit_idx, len(f_freq)-1):
    for k in range(1, fperdecade+1):
        FREQ_TEMPLATE.append(10**(np.log10(f_freq[i]).item()-k/fperdecade))

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
                                if f < 0.1:
                                    ncycle = 1
                                    sample_rate = int(buffer_size / (ncycle / f))
                                    sleep(0.5)
                                elif f <= 100 and f >= 0.1:
                                    ncycle = int(16.2833 * np.log10(f) + 17.4334)
                                    sample_rate = int(buffer_size / (ncycle / f))
                                    sleep(0.5)
                                else:
                                    # est_ncycle = int(0.6228 * np.exp(2.2101*np.log10(f)) * 0.95)
                                    # sample_rate = int(fsample_max)
                                    # ncycle = int(buffer_size/(sample_rate/f))
                                    # while (ncycle < est_ncycle):
                                    #     sample_rate = int(sample_rate * 0.95)
                                    #     ncycle = int(buffer_size/(sample_rate/f))
                                    # if ncycle < 10:
                                    #     ncycle = 10
                                    #     sample_rate = int(f*buffer_size/ncycle)
                                    # sleep(0.5)
                                    ncycle = 50
                                    sample_rate = int(f*buffer_size/ncycle)
                                    sleep(0.5)

                                data_sets = Digi_1.scope_record(sample_rate, buffer_size)
                                print(f"buffer size: {buffer_size}, Perturbation freq: {f}, Sampling frequency: {sample_rate}, Number of cycles: {ncycle}")
                                Digi_1.sendStringUART("STOP")
                            
                            elif res_str == "DoneRecv":
                                # extract csv
                                csv_data = np.column_stack(data_sets)

                                np.savetxt("results/raw_data.csv", 
                                            csv_data, 
                                            delimiter=",", 
                                            header="Current (A),Voltage_B1(V),Voltage_B2(V),Voltage_B3(V)", 
                                            comments="", 
                                            fmt="%.6f")

                                # Calculation Logic
                                Imeas = (data_sets[0] - np.mean(data_sets[0])) / 0.033
                                V1meas = data_sets[1] - np.mean(data_sets[1])
                                V2meas = data_sets[2] - np.mean(data_sets[2])
                                V3meas = data_sets[3] - np.mean(data_sets[3])
                                
                                Imeas_filtered = fir_bandpass(Imeas, sample_rate, f*0.8, f*1.2)
                                V1meas_filtered = fir_bandpass(V1meas, sample_rate, f*0.8, f*1.2)
                                V2meas_filtered = fir_bandpass(V2meas, sample_rate, f*0.8, f*1.2)
                                V3meas_filtered = fir_bandpass(V3meas, sample_rate, f*0.8, f*1.2)
                                
                                # Imeas, _ = remove_baseline_full(data_sets[0] / 0.033, sample_rate, f)
                                # V1meas, _ = remove_baseline_full(data_sets[1], sample_rate, f)
                                # V2meas, _ = remove_baseline_full(data_sets[2], sample_rate, f)
                                # V3meas, _ = remove_baseline_full(data_sets[3], sample_rate, f)

                                # Imeas_filtered = Imeas
                                # V1meas_filtered = V1meas
                                # V2meas_filtered = V2meas
                                # V3meas_filtered = V3meas

                                # extract csv
                                csv_data = np.column_stack([Imeas_filtered, V1meas_filtered, V2meas_filtered, V3meas_filtered])

                                np.savetxt("results/filtered_data.csv", 
                                            csv_data,
                                            delimiter=",", 
                                            header="Current (A),Voltage_B1(V),Voltage_B2(V),Voltage_B3(V)", 
                                            comments="", 
                                            fmt="%.6f")

                                buffer_size = Imeas.shape[0]

                                rng_int = 1 / 10 ** int(-np.log10(f) + 3)

                                if rng_int < 0.001:
                                    I_freq = freq_selection_signal(Imeas_filtered, freq_sweep=[f*0.998, f*1.002], sample_rate=sample_rate)
                                else:
                                    _, _, _, _, I_freq = FFT(Imeas_filtered, freq_sweep=[f*(1-rng_int), f*(1+rng_int)], sample_rate=sample_rate)

                                sfreq = I_freq if I_freq is not None else f
                                
                                Iamp, Iphase = dual_phase_demod(Imeas_filtered, sfreq, sample_rate)
                                V1amp, V1phase = dual_phase_demod(V1meas_filtered, sfreq, sample_rate)
                                V2amp, V2phase = dual_phase_demod(V2meas_filtered, sfreq, sample_rate)
                                V3amp, V3phase = dual_phase_demod(V3meas_filtered, sfreq, sample_rate)

                                print(f"Freq: {sfreq:.5f} Hz | V_amp: {V2amp:.2E} | I_amp: {Iamp:.2E}")

                                I_real = Iamp * np.cos(Iphase+np.pi)
                                I_imag = Iamp * np.sin(Iphase+np.pi)
                                V1_real = V1amp * np.cos(V1phase)
                                V1_imag = V1amp * np.sin(V1phase)
                                V2_real = V2amp * np.cos(V2phase)
                                V2_imag = V2amp * np.sin(V2phase)
                                V3_real = V3amp * np.cos(V3phase)
                                V3_imag = V3amp * np.sin(V3phase)

                                V1_comp = V1_real + 1j * V1_imag
                                V2_comp = V2_real + 1j * V2_imag
                                V3_comp = V3_real + 1j * V3_imag
                                I_comp = I_real + 1j * I_imag
                                Z1 = (V1_comp / I_comp)
                                Z2 = (V2_comp / I_comp)
                                Z3 = (V3_comp / I_comp)

                                # Z1real, Z1imag = calibrator_c1.correct(sfreq, Z1.real, -Z1.imag)
                                # print(f"Cell-1 Impedance: {Z1real} + ({Z1imag}j)")

                                # Z2real, Z2imag = calibrator_c2.correct(sfreq, Z2.real, -Z2.imag)
                                # print(f"Cell-2 Impedance: {Z2real} + ({Z2imag}j)")

                                # Z3real, Z3imag = calibrator_c3.correct(sfreq, Z3.real, -Z3.imag)
                                # print(f"Cell-3 Impedance: {Z3real} + ({Z3imag}j)")

                                Z1real, Z1imag = Z1.real, -Z1.imag
                                Z2real, Z2imag = Z2.real, -Z2.imag
                                Z3real, Z3imag = Z3.real, -Z3.imag

                                # Data Quality Check
                                if i_idx > 0 and ((Z1real < 0.98*sample_c1[i_idx-1, 1] and Z1real < 0) or (Z2real < 0.98*sample_c2[i_idx-1, 1] and Z2real < 0) or (Z3real < 0.98*sample_c3[i_idx-1, 1] and Z3real < 0)):
                                    print("\nFrequency skipped (Impedance Drop)...\n")
                                    break
                                
                                sample_c1[i_idx, 0] = np.mean(data_sets[1])
                                sample_c1[i_idx, 1] = np.log10(sfreq)
                                sample_c1[i_idx, 2] = Z1real
                                sample_c1[i_idx, 3] = Z1imag
                                sample_c1[i_idx, 4] = np.abs(Z1real - 1j * Z1imag)
                                sample_c1[i_idx, 5] = np.angle(Z1real - 1j * Z1imag, deg=True)

                                sample_c2[i_idx, 0] = np.mean(data_sets[2])
                                sample_c2[i_idx, 1] = np.log10(sfreq)
                                sample_c2[i_idx, 2] = Z2real
                                sample_c2[i_idx, 3] = Z2imag
                                sample_c2[i_idx, 4] = np.abs(Z2real - 1j * Z2imag)
                                sample_c2[i_idx, 5] = np.angle(Z2real - 1j * Z2imag, deg=True)

                                sample_c3[i_idx, 0] = np.mean(data_sets[3])
                                sample_c3[i_idx, 1] = np.log10(sfreq)
                                sample_c3[i_idx, 2] = Z3real
                                sample_c3[i_idx, 3] = Z3imag
                                sample_c3[i_idx, 4] = np.abs(Z3real - 1j * Z3imag)
                                sample_c3[i_idx, 5] = np.angle(Z3real - 1j * Z3imag, deg=True)

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
                                sleep(int(3*(3-np.log10(sfreq))))

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