from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod, FFT, fir_bandpass, HolderCalibrator, smooth_impedance_array
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
    [10.0,      0.035723, 0.004255],
    [7.943282,  0.035965, 0.004924],
    [6.309573,  0.036237, 0.005916],
    [5.011872,  0.036878, 0.007005],
    [3.981071,  0.037576, 0.008504],
    [3.162277,  0.038392, 0.010103],
    [2.511886,  0.039571, 0.01187],
    [1.995262,  0.041315, 0.013489],
    [1.584893,  0.042292, 0.01586],
    [1.258925,  0.045037, 0.019321],
    [1.0,       0.048211, 0.021256],
    [0.794328,  0.051981, 0.024925],
    [0.630957,  0.057193, 0.026821],
    [0.501187,  0.062807, 0.028841],
    [0.398107,  0.068942, 0.029241],
    [0.316228,  0.075079, 0.028811],
    [0.251189,  0.080677, 0.027313],
    [0.199526,  0.085651, 0.025091],
    [0.158489,  0.089723, 0.022708],
    [0.125892,  0.092925, 0.020289],
    [0.1,       0.095453, 0.017932],
    [0.079433,  0.097544, 0.015901],
    [0.063096,  0.099351, 0.014283],
    [0.050119,  0.100911, 0.013032],
    [0.039811,  0.102431, 0.012093],
    [0.031623,  0.104094, 0.011573],
    [0.025119,  0.105736, 0.011561],
    [0.019953,  0.107273, 0.011909],
    [0.015849,  0.108798, 0.012386],
    [0.012589,  0.110575, 0.012983],
    [0.01,      0.112905, 0.013835]
])
_ref2 = np.array([
    [10.0,      0.035603, 0.004178],
    [7.943282,  0.03589,  0.004824],
    [6.309573,  0.036185, 0.005712],
    [5.011872,  0.036827, 0.006781],
    [3.981071,  0.037489, 0.00813],
    [3.162277,  0.038331, 0.00959],
    [2.511886,  0.039533, 0.01127],
    [1.995262,  0.041333, 0.012892],
    [1.584893,  0.042702, 0.014784],
    [1.258925,  0.045281, 0.017581],
    [1.0,       0.048542, 0.019183],
    [0.794328,  0.052237, 0.021648],
    [0.630957,  0.056823, 0.022653],
    [0.501187,  0.061644, 0.023506],
    [0.398107,  0.066439, 0.023091],
    [0.316228,  0.071044, 0.022088],
    [0.251189,  0.074985, 0.020458],
    [0.199526,  0.07839,  0.018469],
    [0.158489,  0.081105, 0.016582],
    [0.125892,  0.083209, 0.014803],
    [0.1,       0.084871, 0.013164],
    [0.079433,  0.086291, 0.011785],
    [0.063096,  0.08759,  0.010781],
    [0.050119,  0.088718, 0.010063],
    [0.039811,  0.089863, 0.009559],
    [0.031623,  0.091145, 0.009361],
    [0.025119,  0.092474, 0.009576],
    [0.019953,  0.093726, 0.010111],
    [0.015849,  0.09492,  0.010726],
    [0.012589,  0.096369, 0.011397],
    [0.01,      0.098293, 0.01228]
])
_ref3 = np.array([
    [10.0,      0.038057, 0.004752],
    [7.943282,  0.038411, 0.005568],
    [6.309573,  0.03869,  0.006491],
    [5.011872,  0.039346, 0.007771],
    [3.981071,  0.039952, 0.00932],
    [3.162277,  0.040766, 0.011038],
    [2.511886,  0.041925, 0.013038],
    [1.995262,  0.043721, 0.015188],
    [1.584893,  0.044935, 0.017666],
    [1.258925,  0.047425, 0.021565],
    [1.0,       0.050815, 0.024327],
    [0.794328,  0.054696, 0.028736],
    [0.630957,  0.060208, 0.031784],
    [0.501187,  0.066537, 0.035226],
    [0.398107,  0.07394,  0.036901],
    [0.316228,  0.081908, 0.037734],
    [0.251189,  0.089577, 0.036933],
    [0.199526,  0.096824, 0.03483],
    [0.158489,  0.103069, 0.032051],
    [0.125892,  0.108047, 0.028871],
    [0.1,       0.112028, 0.02554],
    [0.079433,  0.115246, 0.022553],
    [0.063096,  0.117785, 0.020035],
    [0.050119,  0.119851, 0.017994],
    [0.039811,  0.121608, 0.016364],
    [0.031623,  0.123353, 0.015174],
    [0.025119,  0.125079, 0.014661],
    [0.019953,  0.126601, 0.014629],
    [0.015849,  0.128091, 0.014866],
    [0.012589,  0.129718, 0.015274],
    [0.01,      0.131736, 0.016006]
])

# Format: [Freq, Zreal, Zimag]
_meas_holder1 = np.array([
    [10.0,       0.092799, 0.004545],
    [7.943146,   0.093168, 0.005878],
    [6.309448,   0.094303, 0.006889],
    [5.011597,   0.094892, 0.007535],
    [3.980743,   0.095303, 0.008813],
    [3.162109,   0.096601, 0.010423],
    [2.511749,   0.098448, 0.011903],
    [1.995239,   0.099945, 0.014806],
    [1.584656,   0.102504, 0.015951],
    [1.25882,    0.105177, 0.017919],
    [0.999756,   0.109056, 0.019854],
    [0.79425,    0.112609, 0.019939],
    [0.630768,   0.117135, 0.020475],
    [0.501099,   0.120491, 0.018497],
    [0.39798,    0.12408,  0.017859],
    [0.316162,   0.127829, 0.016281],
    [0.250977,   0.129595, 0.016346],
    [0.199524,   0.132484, 0.012615],
    [0.158386,   0.133213, 0.011737],
    [0.125885,   0.134511, 0.009456],
    [0.099875,   0.134908, 0.008365],
    [0.079591,   0.136511, 0.00811],
    [0.063221,   0.138139, 0.005769],
    [0.050219,   0.138326, 0.007944],
    [0.03989,    0.13812,  0.008159],
    [0.031631,   0.138768, 0.006403],
    [0.025169,   0.139221, 0.006207],
    [0.019992,   0.138073, 0.007559],
    [0.015817,   0.138644, 0.007387],
    [0.012614,   0.140521, 0.010287],
    [0.00998,    0.139705, 0.008453]
])
_meas_holder2 = np.array([
    [10.0,       0.0916,   0.004441],
    [7.943146,   0.0919,   0.005626],
    [6.309448,   0.092776, 0.006677],
    [5.011597,   0.093537, 0.007346],
    [3.980743,   0.093924, 0.008626],
    [3.162109,   0.095323, 0.010192],
    [2.511749,   0.097193, 0.011809],
    [1.995239,   0.098549, 0.014358],
    [1.584656,   0.101179, 0.015096],
    [1.25882,    0.1038,   0.016826],
    [0.999756,   0.10747,  0.018489],
    [0.79425,    0.110895, 0.018258],
    [0.630768,   0.115069, 0.01871],
    [0.501099,   0.117889, 0.016648],
    [0.39798,    0.120854, 0.016121],
    [0.316162,   0.124357, 0.014623],
    [0.250977,   0.125672, 0.014531],
    [0.199524,   0.127865, 0.011249],
    [0.158386,   0.128865, 0.01025],
    [0.125885,   0.129693, 0.008288],
    [0.099875,   0.13032,  0.007372],
    [0.079591,   0.131445, 0.008206],
    [0.063221,   0.133153, 0.005239],
    [0.050219,   0.133527, 0.006985],
    [0.03989,    0.133239, 0.007708],
    [0.031631,   0.133663, 0.005852],
    [0.025169,   0.134181, 0.006545],
    [0.019992,   0.133238, 0.007321],
    [0.015817,   0.134073, 0.007203],
    [0.012614,   0.134668, 0.009989],
    [0.00998,    0.134672, 0.008263]
])
_meas_holder3 = np.array([
    [10.0,       0.115536, 0.00481],
    [7.943146,   0.116054, 0.006371],
    [6.309448,   0.117106, 0.007332],
    [5.011597,   0.117861, 0.007862],
    [3.980743,   0.118181, 0.009022],
    [3.162109,   0.119831, 0.010653],
    [2.511749,   0.121975, 0.012047],
    [1.995239,   0.123388, 0.014757],
    [1.584656,   0.126018, 0.015544],
    [1.25882,    0.128463, 0.017352],
    [0.999756,   0.132402, 0.01893],
    [0.79425,    0.135693, 0.01832],
    [0.630768,   0.140065, 0.018874],
    [0.501099,   0.142699, 0.016615],
    [0.39798,    0.145873, 0.016171],
    [0.316162,   0.149421, 0.014822],
    [0.250977,   0.15084,  0.015199],
    [0.199524,   0.153291, 0.011726],
    [0.158386,   0.154295, 0.010849],
    [0.125885,   0.155387, 0.008409],
    [0.099875,   0.155538, 0.00769],
    [0.079591,   0.157621, 0.008286],
    [0.063221,   0.158935, 0.005479],
    [0.050219,   0.159598, 0.007628],
    [0.03989,    0.15902,  0.00837],
    [0.031631,   0.16005,  0.006978],
    [0.025169,   0.160194, 0.006951],
    [0.019992,   0.159219, 0.00792],
    [0.015817,   0.160046, 0.008018],
    [0.012614,   0.161469, 0.011308],
    [0.00998,    0.161534, 0.009333]
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
f_freq = [10e3, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
finit_idx = 3
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
                    sample_c1 = np.zeros([31, 6])
                    sample_c2 = np.zeros([31, 6])
                    sample_c3 = np.zeros([31, 6])
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
                                sample_rate = int(fsample_max)
                                ncycle = int(buffer_size/(sample_rate/f))
                                if f < 0.1:
                                    ncycle = 2
                                    sample_rate = int(buffer_size / (ncycle / f))
                                    
                                elif f <= 10 and f >= 0.1:
                                    ncycle = int(7.5*np.log10(f)+12.5)
                                    sample_rate = int(buffer_size / (ncycle / f))

                                else:
                                    est_ncycle = int(0.6228 * np.exp(2.2101*np.log10(f)))
                                    while (ncycle < est_ncycle):
                                        sample_rate = int(sample_rate * 0.9)
                                        ncycle = int(buffer_size/(sample_rate/f))
                                    if ncycle < 2:
                                        ncycle = 2
                                        sample_rate = int(f*buffer_size/ncycle)
                                data_sets = Digi_1.scope_record(sample_rate, buffer_size)
                                print(f"buffer size: {buffer_size}, Perturbation freq: {f}, Sampling frequency: {sample_rate}, Number of cycles: {ncycle}")

                            elif res_str == "DoneRecv":
                                # Calculation Logic
                                Imeas = (data_sets[0]-np.mean(data_sets[0]))/0.033
                                V1meas = data_sets[1]-np.mean(data_sets[1])
                                V2meas = data_sets[2]-np.mean(data_sets[2])
                                V3meas = data_sets[3]-np.mean(data_sets[3])

                                Imeas_filtered = fir_bandpass(Imeas, sample_rate, f*0.8, f*1.2)
                                V1meas_filtered = fir_bandpass(V1meas, sample_rate, f*0.8, f*1.2)
                                V2meas_filtered = fir_bandpass(V2meas, sample_rate, f*0.8, f*1.2)
                                V3meas_filtered = fir_bandpass(V3meas, sample_rate, f*0.8, f*1.2)

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

                                Z1real, Z1imag = calibrator_c1.correct(sfreq, Z1.real, -Z1.imag)
                                print(f"Cell-1 Impedance: {Z1real} + ({Z1imag}j)")

                                Z2real, Z2imag = calibrator_c2.correct(sfreq, Z2.real, -Z2.imag)
                                print(f"Cell-2 Impedance: {Z2real} + ({Z2imag}j)")

                                Z3real, Z3imag = calibrator_c3.correct(sfreq, Z3.real, -Z3.imag)
                                print(f"Cell-3 Impedance: {Z3real} + ({Z3imag}j)")

                                # Z1real, Z1imag = Z1.real, -Z1.imag
                                # Z2real, Z2imag = Z2.real, -Z2.imag
                                # Z3real, Z3imag = Z3.real, -Z3.imag

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
                                output_c1 = SoH_est.predict(sample_c1.reshape(1, 6, 31).astype(np.float32))
                                print(f"\n\nThe estimated SoH of cell-1 is: {str(np.round(output_c1*100, decimals=2))}%\n")
                                output_c2 = SoH_est.predict(sample_c2.reshape(1, 6, 31).astype(np.float32))
                                print(f"\n\nThe estimated SoH of cell-2 is: {str(np.round(output_c2*100, decimals=2))}%\n")
                                output_c3 = SoH_est.predict(sample_c3.reshape(1, 6, 31).astype(np.float32))
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