from MyDigilent import MyDigilent, freq_selection_signal, dual_phase_demod
from time import sleep
import numpy as np, socket, struct
import select # Helper for checking socket status

# --- TCP Configuration ---
TCP_IP = '10.115.78.142'  
TCP_PORT = 5005           

print(f"Connecting to server at {TCP_IP}:{TCP_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((TCP_IP, TCP_PORT))
    print("Connected! Entering Standby Mode.")
except ConnectionRefusedError:
    print("Failed to connect. Is the Receiver script running?")
    quit()

# --- Hardware Initialization ---
# UART specs
PIN_TX = 0           
PIN_RX = 1           
BAUDRATE = 115200

# Device object generation
Digi_1 = MyDigilent(tx=PIN_TX, rx=PIN_RX, baud_rate=BAUDRATE, parity="none", data_bits=8, stop_bits=1)
Digi_1.scope_setup(channels=[1, 2])
sleep(1)

# --- Frequency Setup ---
f_freq = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
finit_idx = 1
fperdecade = 10
FREQ_TEMPLATE = [] # Renamed to template so we can copy it for each run
FREQ_TEMPLATE.append(f_freq[finit_idx])
for i in range(finit_idx, finit_idx+4):
    for k in range(1, fperdecade+1):
        FREQ_TEMPLATE.append(10**(np.log10(f_freq[i]).item()-k/fperdecade))

# --- Main Standby Loop ---
try:
    while True:
        # 1. STANDBY STATE: Wait for "START" command
        print("\n--- STANDBY: Waiting for 'START' command ---")
        
        # Ensure socket is in blocking mode while waiting
        sock.setblocking(True) 
        
        # Receive command (buffers up to 1024 bytes)
        try:
            command_raw = sock.recv(1024)
            if not command_raw:
                print("Host closed connection.")
                break # Exit if host kills connection
            
            command = command_raw.decode('utf-8').strip()
        except Exception as e:
            print(f"Connection Error: {e}")
            break

        if command == "START":
            print("START received. Beginning Measurement Sequence...")
            
            # 2. RUNNING STATE: Initialize variables for this run
            sample = np.zeros([61, 3])
            i_idx = 0
            stop_requested = False

            # Loop through frequencies
            for f in FREQ_TEMPLATE:
                
                # --- CHECK FOR STOP COMMAND (Non-Blocking) ---
                try:
                    sock.setblocking(False) # Don't wait, just peek
                    # We look for new data. If no data, it raises BlockingIOError
                    cmd_check = sock.recv(1024) 
                    if cmd_check and "STOP" in cmd_check.decode('utf-8'):
                        print("\n!!! STOP command received. Halting measurement !!!")
                        stop_requested = True
                        sock.setblocking(True) # Restore blocking
                        break # Break out of frequency loop
                except BlockingIOError:
                    pass # No command waiting, continue normal execution
                except Exception as e:
                    print(f"Socket error during check: {e}")
                    stop_requested = True
                    break
                
                sock.setblocking(True) # Restore blocking for normal data sending
                # ---------------------------------------------

                CMD = str(f)
                Digi_1.sendStringUART(CMD)
                sleep(1) # Give time for UART
                
                mainloop = True
                while mainloop:
                    # Read UART response
                    RES = bytes(Digi_1.uart_read())
                    res_str = RES.decode("utf-8")

                    if res_str == "Received":
                        print(f"Measuring EIS at {CMD.strip()} Hz...")
                        
                        # Scope configuration based on freq
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
                        if i_idx > 0 and Z.real < 0.95*sample[i_idx-1, 1]:
                            print("\nFrequency skipped (Impedance Drop)...\n")
                            break # Breaks the while(mainloop), continues for loop

                        # Store Data
                        sample[i_idx, 0] = sfreq
                        sample[i_idx, 1] = Z.real
                        sample[i_idx, 2] = -Z.imag
                        
                        # --- Send Data to Host ---
                        try:
                            # Send accumulated buffer
                            data_bytes = sample[i_idx, :].flatten().tobytes()
                            header = struct.pack('>I', len(data_bytes))
                            sock.sendall(header + data_bytes)
                            print(f"Sent update to host.")
                        except Exception as e:
                            print(f"Send failed: {e}")

                        i_idx += 1
                        mainloop = False # Exit UART wait loop

            # --- End of Run or Stopped ---
            print("Sequence finished or stopped. Saving data...")
            
            # Filter empty rows
            rows_to_keep = ~ (sample == 0).all(axis=1)
            final_sample = sample[rows_to_keep]

            # Save to CSV
            # filename = f"EIS_Data/Test_Run_{int(np.random.rand()*1000)}.csv" # Random ID to prevent overwrite
            # np.savetxt(filename, final_sample, delimiter=',')
            # print(f"Data saved to {filename}")
            
            # Loop restarts, going back to 'WAITING FOR START'
            
except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected.")

finally:
    Digi_1.close()
    sock.close()
    print("Device and Socket closed.")