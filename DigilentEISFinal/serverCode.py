from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod
from time import sleep
import numpy as np, socket, struct
import select

# --- TCP Configuration ---
TCP_IP = '10.115.78.142'  
TCP_PORT = 5005           

# --- Hardware Initialization ---
# UART specs
PIN_TX = 0           
PIN_RX = 1           
BAUDRATE = 115200

# Device object generation
# We initialize this ONCE so we don't reset the hardware every time a client connects
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
print(f"Server started on {TCP_IP}:{TCP_PORT}")

# Create the Master Socket (The one that listens)
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow instant restart
try:
    server_sock.bind((TCP_IP, TCP_PORT))
    server_sock.listen(1)
except Exception as e:
    print(f"Critical Server Error: {e}")
    quit()

try:
    while True:
        print("\n--- STANDBY: Waiting for Host Connection... ---")
        
        # 1. Accept new connection (BLOCKING)
        conn, addr = server_sock.accept()
        print(f"Client connected from: {addr}")
        
        # Flag to track if the current client is active
        client_connected = True

        while client_connected:
            try:
                # 2. Wait for "START" command
                # We use select to check for data/disconnection without blocking forever
                ready_to_read, _, _ = select.select([conn], [], [], 1.0)
                
                if ready_to_read:
                    command_raw = conn.recv(1024)
                    if not command_raw:
                        print("Client disconnected.")
                        client_connected = False
                        break 
                    
                    command = command_raw.decode('utf-8').strip()
                    
                    if command == "START":
                        print("START received. Beginning Sequence...")
                        
                        # --- MEASUREMENT LOOP ---
                        sample = np.zeros([61, 3])
                        i_idx = 0
                        
                        for f in FREQ_TEMPLATE:
                            # A. Check for STOP or Disconnect during loop
                            # Use non-blocking peek
                            conn.setblocking(False)
                            try:
                                cmd_check = conn.recv(1024)
                                if not cmd_check: # Empty bytes = Disconnect
                                    print("Client disconnected during sweep.")
                                    client_connected = False
                                    break
                                if "STOP" in cmd_check.decode('utf-8'):
                                    print("STOP command received.")
                                    break
                            except BlockingIOError:
                                pass # No data waiting, carry on
                            except Exception as e:
                                print(f"Connection lost: {e}")
                                client_connected = False
                                break
                            finally:
                                conn.setblocking(True)

                            if not client_connected: break

                            # B. Perform Measurement
                            CMD = str(f)
                            Digi_1.sendStringUART(CMD)
                            sleep(0.5) 
                            
                            mainloop = True
                            while mainloop:
                                RES = bytes(Digi_1.uart_read())
                                res_str = RES.decode("utf-8")

                                if res_str == "Received":
                                    # Setup Scope logic...
                                    if f < 1: buffer_size = 180
                                    elif 1 <= f <= 5: buffer_size = 450
                                    else: buffer_size = 900
                                    sample_rate = int(100*float(CMD))
                                    data_sets = Digi_1.scope_record(sample_rate, buffer_size)

                                elif res_str == "DoneRecv":
                                    # --- Processing Logic ---
                                    Imeas = 100*(data_sets[0]-np.mean(data_sets[0]))
                                    V1meas = data_sets[1]-np.mean(data_sets[1])

                                    I_freq = freq_selection_signal(Imeas, freq_sweep=[f*0.9, f*1.1], sample_rate=sample_rate)
                                    
                                    sfreq = I_freq if I_freq is not None else f
                                    
                                    Iamp, Iphase = dual_phase_demod(Imeas, sfreq, sample_rate)
                                    V1amp, V1phase = dual_phase_demod(V1meas, sfreq, sample_rate)

                                    print(f"Freq: {sfreq:.2f} Hz | V_amp: {V1amp:.2E} | I_amp: {Iamp:.2E}")

                                    # Complex Calculations
                                    I_real = Iamp * np.cos(Iphase+np.pi)
                                    I_imag = Iamp * np.sin(Iphase+np.pi)
                                    V1_real = V1amp * np.cos(V1phase)
                                    V1_imag = V1amp * np.sin(V1phase)

                                    V_comp = V1_real + 1j * V1_imag
                                    I_comp = I_real + 1j * I_imag
                                    Z = (V_comp / I_comp)

                                    print("Impedance: " + str(Z.real) + "+(" + str(Z.imag) + "j)")

                                    # Data Quality Check
                                    if i_idx > 0 and (Z.real < 0.95*sample[i_idx-1, 1] or Z.real < 1e-7):
                                        print("\nFrequency skipped (Impedance Drop)...\n")
                                        break # Breaks the while(mainloop), continues for loop

                                    # Store Data
                                    sample[i_idx, 0] = sfreq
                                    sample[i_idx, 1] = Z.real
                                    sample[i_idx, 2] = -Z.imag

                                    # C. Send Data (Protects against broken pipe)
                                    try:
                                        data_bytes = sample[i_idx, :].flatten().tobytes()
                                        header = struct.pack('>I', len(data_bytes))
                                        conn.sendall(header + data_bytes)
                                    except (BrokenPipeError, ConnectionResetError):
                                        print("Client disconnected while sending data.")
                                        client_connected = False
                                        break
                                    
                                    mainloop = False # Exit UART wait
                        
                        print("Sweep ended (Finished or Stopped).")
                        # (Optional) Save data to CSV locally on ADP3450 here if desired
                        
            except (ConnectionResetError, BrokenPipeError):
                print("Connection lost.")
                client_connected = False
            except Exception as e:
                print(f"Unexpected Error: {e}")
                client_connected = False
        
        # Cleanup the specific client connection
        if conn: conn.close()
        print("Client session ended. Returning to Standby.\n")

except KeyboardInterrupt:
    print("\nServer shutting down manually.")
finally:
    Digi_1.close()
    server_sock.close()