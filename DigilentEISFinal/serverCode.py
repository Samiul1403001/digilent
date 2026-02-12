from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod
from time import sleep
import numpy as np, socket, struct
import select 

# --- TCP Configuration ---
# 1. SERVER CONFIG: Listen on ALL interfaces
TCP_IP = '0.0.0.0'
TCP_PORT = 5005            

# --- Hardware Initialization ---
PIN_TX = 0            
PIN_RX = 1            
BAUDRATE = 115200

# Initialize Hardware ONCE (before entering the network loop)
print("Initializing Digilent-ADP3450...")
Digi_1 = MyDigilent(tx=PIN_TX, rx=PIN_RX, baud_rate=BAUDRATE, parity="none", data_bits=8, stop_bits=1)
Digi_1.scope_setup(channels=[1, 2])
sleep(1)

# --- Frequency Setup ---
f_freq = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
finit_idx = 1
fperdecade = 10
FREQ_TEMPLATE = [] 
FREQ_TEMPLATE.append(f_freq[finit_idx])
for i in range(finit_idx, finit_idx+4):
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
                    sample = np.zeros([61, 3])
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
                                if f < 1: buffer_size = 400
                                elif 1 <= f <= 10: buffer_size = 2600
                                else: buffer_size = 11000
                                sample_rate = int(200*float(CMD))
                                data_sets = Digi_1.scope_record(sample_rate, buffer_size)

                            elif res_str == "DoneRecv":
                                # Calculation Logic
                                Imeas = 100*(data_sets[0]-np.mean(data_sets[0]))
                                V1meas = data_sets[1]-np.mean(data_sets[1])

                                I_freq = freq_selection_signal(Imeas, freq_sweep=[f*0.9, f*1.1], sample_rate=sample_rate)
                                sfreq = I_freq if I_freq is not None else f
                                
                                Iamp, Iphase = dual_phase_demod(Imeas, sfreq, sample_rate)
                                V1amp, V1phase = dual_phase_demod(V1meas, sfreq, sample_rate)

                                print(f"Freq: {sfreq:.2f} Hz | V_amp: {V1amp:.2E} | I_amp: {Iamp:.2E}")

                                I_real = Iamp * np.cos(Iphase+np.pi)
                                I_imag = Iamp * np.sin(Iphase+np.pi)
                                V1_real = V1amp * np.cos(V1phase)
                                V1_imag = V1amp * np.sin(V1phase)

                                V_comp = V1_real + 1j * V1_imag
                                I_comp = I_real + 1j * I_imag
                                Z = (V_comp / I_comp)

                                print("Impedance: " + str(Z.real) + "+(" + str(Z.imag) + "j)")

                                # Data Quality Check
                                if i_idx > 0 and Z.real < 0.98*sample[i_idx-1, 1] and Z.real < 0:
                                    print("\nFrequency skipped (Impedance Drop)...\n")
                                    break 

                                sample[i_idx, 0] = sfreq
                                sample[i_idx, 1] = Z.real
                                sample[i_idx, 2] = -Z.imag
                                
                                # --- Send Data to Host ---
                                try:
                                    # Send CURRENT sample row (3 floats)
                                    data_bytes = sample[i_idx, :].flatten().tobytes()
                                    header = struct.pack('>I', len(data_bytes))
                                    conn.sendall(header + data_bytes)
                                    print(f"Sent measurement to Client.")
                                except Exception as e:
                                    print(f"Send failed (Client disconnected?): {e}")
                                    client_connected = False
                                    stop_requested = True # Force loop exit

                                i_idx += 1
                                mainloop = False 

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