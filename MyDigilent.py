import ctypes, time               # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators
import inspect, numpy as np       # caller function data
import dwfconstants as constants

def clean_buffer(y_buffer, signal_freq, sample_rate):
    # 1. Synthesize the Time Vector based on indices
    # t = [0, 1/fs, 2/fs, ..., N/fs]
    n_samples = len(y_buffer)
    t = np.arange(n_samples) / sample_rate
    
    # 2. Create Design Matrix
    omega = 2 * np.pi * signal_freq
    # Solves for: y = a*sin(wt) + b*cos(wt) + c
    M = np.vstack([
        np.sin(omega * t), 
        np.cos(omega * t), 
        np.ones(n_samples)
    ]).T
    
    # 3. Least Squares Fit
    coeffs, _, _, _ = np.linalg.lstsq(M, y_buffer, rcond=None)
    a, b, c = coeffs
    
    # 4. Reconstruct Clean Signal
    y_clean = a * np.sin(omega * t) + b * np.cos(omega * t) + c
    
    # Optional: Calculate physical parameters relative to index 0
    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)
    
    return y_clean, (amplitude, phase, c)

def freq_selection_signal(y_buffer, freq_sweep, sample_rate):
    mamp = 0
    freq = []
    c = 0
    for f in np.arange(freq_sweep[0], freq_sweep[1], 0.1):
        _, params = clean_buffer(y_buffer, f, sample_rate)
        if params[0] > mamp:
            mamp = params[0]
            freq.append(f)
        else:
            c += 1
            if c >= 2:
                break
    return freq[-1]

class data:
    """ stores the device handle, the device name and the device data """
    handle = ctypes.c_int(0)
    name = ""
    version = ""
    class analog:
        class input:
            channel_count = 0
            max_buffer_size = 0
            max_resolution = 0
            min_range = 0
            max_range = 0
            steps_range = 0
            min_offset = 0
            max_offset = 0
            steps_offset = 0
        class output:
            channel_count = 0
            node_count = []
            node_type = []
            max_buffer_size = []
            min_amplitude = []
            max_amplitude = []
            min_offset = []
            max_offset = []
            min_frequency = []
            max_frequency = []
        class IO:
            channel_count = 0
            node_count = []
            channel_name = []
            channel_label = []
            node_name = []
            node_unit = []
            min_set_range = []
            max_set_range = []
            min_read_range = []
            max_read_range = []
            set_steps = []
            read_steps = []
    class digital:
        class input:
            channel_count = 0
            max_buffer_size = 0
        class output:
            channel_count = 0
            max_buffer_size = 0

class error(Exception):
    """
        WaveForms SDK error
    """
    def __init__(self, message, function, instrument):
        self.message = message
        self.function = function
        self.instrument = instrument
        return
    def __str__(self):
        return "Error: " + self.instrument + " -> " + self.function + " -> " + self.message

class warning(Exception):
    """
        WaveForms SDK warning, or non-fatal error
    """
    def __init__(self, message, function, instrument):
        self.message = message
        self.function = function
        self.instrument = instrument
        return
    def __str__(self):
        return "Warning: " + self.instrument + " -> " + self.function + " -> " + self.message

class MyDigilent:
    def __init__(self, rx, tx, baud_rate=115200, parity=None, data_bits=8, stop_bits=1):
        self.device = None
        self.config = 0

        # load the dynamic library, get constants path (the path is OS specific)
        if platform.startswith("win"):
            # on Windows
            self.dwf = ctypes.cdll.dwf
            constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"
        elif platform.startswith("darwin"):
            # on macOS
            lib_path = sep + "Library" + sep + "Frameworks" + sep + "dwf.framework" + sep + "dwf"
            self.dwf = ctypes.cdll.LoadLibrary(lib_path)
            constants_path = sep + "Applications" + sep + "WaveForms.app" + sep + "Contents" + sep + "Resources" + sep + "SDK" + sep + "samples" + sep + "py"
        else:
            # on Linux
            self.dwf = ctypes.cdll.LoadLibrary("libdwf.so")
            constants_path = sep + "usr" + sep + "share" + sep + "digilent" + sep + "waveforms" + sep + "samples" + sep + "py"

        # import constants
        path.append(constants_path)

        device_names = [("Analog Discovery", constants.devidDiscovery), ("Analog Discovery 2", constants.devidDiscovery2),
                    ("Analog Discovery Studio", constants.devidDiscovery2), ("Digital Discovery", constants.devidDDiscovery),
                    ("Analog Discovery Pro 3X50", constants.devidADP3X50), ("Analog Discovery Pro 5250", constants.devidADP5250)]
    
        # decode device names
        device_type = constants.enumfilterAll
        for pair in device_names:
            if pair[0] == self.device:
                device_type = pair[1]
                break

        # count devices
        device_count = ctypes.c_int()
        self.dwf.FDwfEnum(device_type, ctypes.byref(device_count))

        # check for connected devices
        if device_count.value <= 0:
            if device_type.value == 0:
                raise error("There are no connected devices", "open", "device")
            else:
                raise error("Error: There is no " + str(self.device) + " connected", "open", "device")

        # this is the device handle - it will be used by all functions to "address" the connected device
        device_handle = ctypes.c_int(0)

        # connect to the first available device
        index = 0
        while device_handle.value == 0 and index < device_count.value:
            self.dwf.FDwfDeviceConfigOpen(ctypes.c_int(index), ctypes.c_int(self.config), ctypes.byref(device_handle))
            index += 1  # increment the index and try again if the device is busy

        # check connected device type
        device_name = ""
        if device_handle.value != 0:
            device_id = ctypes.c_int()
            device_rev = ctypes.c_int()
            self.dwf.FDwfEnumDeviceType(ctypes.c_int(index - 1), ctypes.byref(device_id), ctypes.byref(device_rev))

            # decode device id
            for pair in device_names:
                if pair[1].value == device_id.value:
                    device_name = pair[0]
                    break

        # check for errors
        # if the device handle is empty after a connection attempt
        if device_handle == constants.hdwfNone:
            # check for errors
            err_nr = ctypes.c_int() # variable for error number
            self.dwf.FDwfGetLastError(ctypes.byref(err_nr));  # get error number
            # if there is an error
            if err_nr != constants.dwfercNoErc:
                # check the error message
                self.check_error()
        global data
        data.handle = device_handle
        data.name = device_name
        self.dev = self.__get_info__(data)

        """
        initializes UART communication
        
        parameters: - device data
                    - rx (DIO line used to receive data)
                    - tx (DIO line used to send data)
                    - baud_rate (communication speed, default is 9600 bits/s)
                    - parity possible: None (default), True means even, False means odd
                    - data_bits (default is 8)
                    - stop_bits (default is 1)
        """
        # set baud rate
        if self.dwf.FDwfDigitalUartRateSet(self.dev.handle, ctypes.c_double(baud_rate)) == 0:
            self.check_error()

        # set communication channels
        if self.dwf.FDwfDigitalUartTxSet(self.dev.handle, ctypes.c_int(tx)) == 0:
            self.check_error()
        if self.dwf.FDwfDigitalUartRxSet(self.dev.handle, ctypes.c_int(rx)) == 0:
            self.check_error()

        # set data bit count
        if self.dwf.FDwfDigitalUartBitsSet(self.dev.handle, ctypes.c_int(data_bits)) == 0:
            self.check_error()

        # set parity bit requirements
        if parity == True:
            parity = 2
        elif parity == False:
            parity = 1
        else:
            parity = 0
        if self.dwf.FDwfDigitalUartParitySet(self.dev.handle, ctypes.c_int(parity)) == 0:
            self.check_error()

        # set stop bit count
        if self.dwf.FDwfDigitalUartStopSet(self.dev.handle, ctypes.c_double(stop_bits)) == 0:
            self.check_error()

        # initialize channels with idle levels

        # dummy read
        dummy_buffer = ctypes.create_string_buffer(0)
        dummy_buffer = ctypes.c_int(0)
        dummy_parity_flag = ctypes.c_int(0)
        if self.dwf.FDwfDigitalUartRx(self.dev.handle, dummy_buffer, ctypes.c_int(0), ctypes.byref(dummy_buffer), ctypes.byref(dummy_parity_flag)) == 0:
            self.check_error()

        # dummy write
        if self.dwf.FDwfDigitalUartTx(self.dev.handle, dummy_buffer, ctypes.c_int(0)) == 0:
            self.check_error()

    def uart_read(self):
        """
            receives data from UART
            
            parameters: - device data

            return:     - integer list containing the received bytes
        """
        # variable to store results
        rx_data = []

        # create empty string buffer
        data = (ctypes.c_ubyte * 8193)()

        # character counter
        count = ctypes.c_int(0)

        # parity flag
        parity_flag= ctypes.c_int(0)

        # read up to 8k characters
        if self.dwf.FDwfDigitalUartRx(self.dev.handle, data, ctypes.c_int(ctypes.sizeof(data)-1), ctypes.byref(count), ctypes.byref(parity_flag)) == 0:
            self.check_error()

        # append current data chunks
        for index in range(0, count.value):
            rx_data.append(int(data[index]))

        # ensure data integrity
        while count.value > 0:
            # create empty string buffer
            data = (ctypes.c_ubyte * 8193)()

            # character counter
            count = ctypes.c_int(0)

            # parity flag
            parity_flag= ctypes.c_int(0)

            # read up to 8k characters
            if self.dwf.FDwfDigitalUartRx(self.dev.handle, data, ctypes.c_int(ctypes.sizeof(data)-1), ctypes.byref(count), ctypes.byref(parity_flag)) == 0:
                self.check_error()
            # append current data chunks
            for index in range(0, count.value):
                rx_data.append(int(data[index]))

            # check for not acknowledged
            if parity_flag.value < 0:
                raise warning("Buffer overflow", "read", "protocol/uart")
            elif parity_flag.value > 0:
                raise warning("Parity error: index {}".format(parity_flag.value), "read", "protocol/uart")
        return rx_data

    def uart_write(self, data):
        """
            send data through UART
            
            parameters: - data of type string, int, or list of characters/integers
        """
        # cast data
        if type(data) == int:
            data = "".join(chr(data))
        elif type(data) == list:
            data = "".join(chr(element) for element in data)

        # encode the string into a string buffer
        data = ctypes.create_string_buffer(data.encode("UTF-8"))

        # send text, trim zero ending
        if self.dwf.FDwfDigitalUartTx(self.dev.handle, data, ctypes.c_int(ctypes.sizeof(data)-1)) == 0:
            self.check_error()

        return

    def sendStringUART(self, section):
        i = 0
        while i < 8:
            if i < len(section):
                self.uart_write(section[i])
            else:
                self.uart_write("\0")
            i += 1

    def scope_setup(self, channels=[1, 2]):
        self.channels = channels
        print(f"Configuring {len(self.channels)} channel(s)...")

        # Enable all 4 channels (Indices 0, 1, 2, 3)
        for i in self.channels:
            # Enable channel
            self.dwf.FDwfAnalogInChannelEnableSet(self.dev.handle, ctypes.c_int(i-1), ctypes.c_bool(True))
            # Set Range (e.g., 5V peak-to-peak)
            self.dwf.FDwfAnalogInChannelRangeSet(self.dev.handle, ctypes.c_int(i-1), ctypes.c_double(5.0))
            # Set Offset (0V)
            self.dwf.FDwfAnalogInChannelOffsetSet(self.dev.handle, ctypes.c_int(i-1), ctypes.c_double(0.0))

    def scope_record(self, sample_rate=1e3, buffer_size=300):
        # Set Master Acquisition Parameters
        self.dwf.FDwfAnalogInFrequencySet(self.dev.handle, ctypes.c_double(sample_rate))
        self.dwf.FDwfAnalogInBufferSizeSet(self.dev.handle, ctypes.c_int(buffer_size))

        # 4. Start the Acquisition
        # This single command starts the capture for ALL enabled channels simultaneously.
        # Reconfigure = False, Start = True
        print("Starting acquisition...")
        self.dwf.FDwfAnalogInConfigure(self.dev.handle, ctypes.c_bool(False), ctypes.c_bool(True))

        # 5. Wait for acquisition to finish
        status = ctypes.c_byte()
        while True:
            self.dwf.FDwfAnalogInStatus(self.dev.handle, ctypes.c_bool(True), ctypes.byref(status))
            if status.value == constants.DwfStateDone.value:
                break
            time.sleep(0.01)
        
        print("Acquisition done. Fetching data...")

        # 6. Retrieve Data
        # We create a dictionary or list to store arrays for each channel
        data_sets = []

        # Allocate a C-type double array for the buffer
        c_buffer = (ctypes.c_double * buffer_size)()

        for i in self.channels:
            # Fetch data for channel 'i' from the device to our local 'c_buffer'
            self.dwf.FDwfAnalogInStatusData(self.dev.handle, ctypes.c_int(i-1), c_buffer, ctypes.c_int(buffer_size))
            # Convert to standard Python list/numpy array and store
            data_sets.append(np.array(c_buffer))
        
        return data_sets

    def check_error(self):
        """
            check for errors
        """
        err_msg = ctypes.create_string_buffer(512)        # variable for the error message
        self.dwf.FDwfGetLastErrorMsg(err_msg)                  # get the error message
        err_msg = err_msg.value.decode("ascii")           # format the message
        if err_msg != "":
            err_func = inspect.stack()[1].function        # get caller function
            err_inst = inspect.stack()[1].filename        # get caller file name
            # delete the extension
            err_inst = err_inst.split('.')[0]
            # delete the path
            path_list = err_inst.split('/')
            err_inst = path_list[-1]
            path_list = err_inst.split('\\')
            err_inst = path_list[-1]
            raise error(err_msg, err_func, err_inst)
        return

    def __get_info__(self, device_data):
        """
            get and return device information
        """
        # check WaveForms version
        version = ctypes.create_string_buffer(16)
        if self.dwf.FDwfGetVersion(version) == 0:
            self.check_error()
        device_data.version = str(version.value)[2:-1]

        # define temporal variables
        temp1 = ctypes.c_int()
        temp2 = ctypes.c_int()
        temp3 = ctypes.c_int()

        # analog input information
        # channel count
        if self.dwf.FDwfAnalogInChannelCount(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.analog.input.channel_count = temp1.value
        # buffer size
        if self.dwf.FDwfAnalogInBufferSizeInfo(device_data.handle, 0, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.analog.input.max_buffer_size = temp1.value
        # ADC resolution
        if self.dwf.FDwfAnalogInBitsInfo(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.analog.input.max_resolution = temp1.value
        # range information
        temp1 = ctypes.c_double()
        temp2 = ctypes.c_double()
        temp3 = ctypes.c_double()
        if self.dwf.FDwfAnalogInChannelRangeInfo(device_data.handle, ctypes.byref(temp1), ctypes.byref(temp2), ctypes.byref(temp3)) == 0:
            self.check_error()
        device_data.analog.input.min_range = temp1.value
        device_data.analog.input.max_range = temp2.value
        device_data.analog.input.steps_range = int(temp3.value)
        # offset information
        if self.dwf.FDwfAnalogInChannelOffsetInfo(device_data.handle, ctypes.byref(temp1), ctypes.byref(temp2), ctypes.byref(temp3)) == 0:
            self.check_error()
        device_data.analog.input.min_offset = temp1.value
        device_data.analog.input.max_offset = temp2.value
        device_data.analog.input.steps_offset = int(temp3.value)

        # analog output information
        temp1 = ctypes.c_int()
        if self.dwf.FDwfAnalogOutCount(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.analog.output.channel_count = temp1.value
        for channel_index in range(device_data.analog.output.channel_count):
            # check node types and node count
            temp1 = ctypes.c_int()
            if self.dwf.FDwfAnalogOutNodeInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.byref(temp1)) == 0:
                self.check_error()
            templist = []
            for node_index in range(3):
                if ((1 << node_index) & int(temp1.value)) == 0:
                    continue
                elif node_index == constants.AnalogOutNodeCarrier.value:
                    templist.append("carrier")
                elif node_index == constants.AnalogOutNodeFM.value:
                    templist.append("FM")
                elif node_index == constants.AnalogOutNodeAM.value:
                    templist.append("AM")
            device_data.analog.output.node_type.append(templist)
            device_data.analog.output.node_count.append(len(templist))
            # buffer size
            templist = []
            for node_index in range(device_data.analog.output.node_count[channel_index]):
                if self.dwf.FDwfAnalogOutNodeDataInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), 0, ctypes.byref(temp1)) == 0:
                    self.check_error()
                templist.append(temp1.value)
            device_data.analog.output.max_buffer_size.append(templist)
            # amplitude information
            templist1 = []
            templist2 = []
            temp1 = ctypes.c_double()
            temp2 = ctypes.c_double()
            for node_index in range(device_data.analog.output.node_count[channel_index]):
                if self.dwf.FDwfAnalogOutNodeAmplitudeInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), ctypes.byref(temp1), ctypes.byref(temp2)) == 0:
                    self.check_error()
                templist1.append(temp1.value)
                templist2.append(temp2.value)
            device_data.analog.output.min_amplitude.append(templist1)
            device_data.analog.output.max_amplitude.append(templist2)
            # offset information
            templist1 = []
            templist2 = []
            for node_index in range(device_data.analog.output.node_count[channel_index]):
                if self.dwf.FDwfAnalogOutNodeOffsetInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), ctypes.byref(temp1), ctypes.byref(temp2)) == 0:
                    self.check_error()
                templist1.append(temp1.value)
                templist2.append(temp2.value)
            device_data.analog.output.min_offset.append(templist1)
            device_data.analog.output.max_offset.append(templist2)
            # frequency information
            templist1 = []
            templist2 = []
            for node_index in range(device_data.analog.output.node_count[channel_index]):
                if self.dwf.FDwfAnalogOutNodeFrequencyInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), ctypes.byref(temp1), ctypes.byref(temp2)) == 0:
                    self.check_error()
                templist1.append(temp1.value)
                templist2.append(temp2.value)
            device_data.analog.output.min_frequency.append(templist1)
            device_data.analog.output.max_frequency.append(templist2)

        # analog IO information
        # channel count
        temp1 = ctypes.c_int()
        if self.dwf.FDwfAnalogIOChannelCount(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.analog.IO.channel_count = temp1.value
        for channel_index in range(device_data.analog.IO.channel_count):
            # channel names and labels
            temp1 = ctypes.create_string_buffer(256)
            temp2 = ctypes.create_string_buffer(256)
            if self.dwf.FDwfAnalogIOChannelName(device_data.handle, ctypes.c_int(channel_index), temp1, temp2) == 0:
                self.check_error()
            device_data.analog.IO.channel_name.append(str(temp1.value)[2:-1])
            device_data.analog.IO.channel_label.append(str(temp2.value)[2:-1])
            # check node count
            temp1 = ctypes.c_int()
            if self.dwf.FDwfAnalogIOChannelInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.byref(temp1)) == 0:
                self.check_error()
            device_data.analog.IO.node_count.append(temp1.value)
            # node names and units
            templist1 = []
            templist2 = []
            for node_index in range(device_data.analog.IO.node_count[channel_index]):
                temp1 = ctypes.create_string_buffer(256)
                temp2 = ctypes.create_string_buffer(256)
                if self.dwf.FDwfAnalogIOChannelNodeName(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), temp1, temp2) == 0:
                    self.check_error()
                templist1.append(str(temp1.value)[2:-1])
                templist2.append(str(temp2.value)[2:-1])
            device_data.analog.IO.node_name.append(templist1)
            device_data.analog.IO.node_unit.append(templist2)
            # node write info
            templist1 = []
            templist2 = []
            templist3 = []
            temp1 = ctypes.c_double()
            temp2 = ctypes.c_double()
            temp3 = ctypes.c_int()
            for node_index in range(device_data.analog.IO.node_count[channel_index]):
                if self.dwf.FDwfAnalogIOChannelNodeSetInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), ctypes.byref(temp1), ctypes.byref(temp2), ctypes.byref(temp3)) == 0:
                    self.check_error()
                templist1.append(temp1.value)
                templist2.append(temp2.value)
                templist3.append(temp3.value)
            device_data.analog.IO.min_set_range.append(templist1)
            device_data.analog.IO.max_set_range.append(templist2)
            device_data.analog.IO.set_steps.append(templist3)
            # node read info
            templist1 = []
            templist2 = []
            templist3 = []
            for node_index in range(device_data.analog.IO.node_count[channel_index]):
                if self.dwf.FDwfAnalogIOChannelNodeStatusInfo(device_data.handle, ctypes.c_int(channel_index), ctypes.c_int(node_index), ctypes.byref(temp1), ctypes.byref(temp2), ctypes.byref(temp3)) == 0:
                    self.check_error()
                templist1.append(temp1.value)
                templist2.append(temp2.value)
                templist3.append(temp3.value)
            device_data.analog.IO.min_read_range.append(templist1)
            device_data.analog.IO.max_read_range.append(templist2)
            device_data.analog.IO.read_steps.append(templist3)

        # digital input information
        # channel count
        temp1 = ctypes.c_int()
        if self.dwf.FDwfDigitalInBitsInfo(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.digital.input.channel_count = temp1.value
        # buffer size
        if self.dwf.FDwfDigitalInBufferSizeInfo(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.digital.input.max_buffer_size = temp1.value

        # digital output information
        # channel count
        if self.dwf.FDwfDigitalOutCount(device_data.handle, ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.digital.output.channel_count = temp1.value
        # buffer size
        if self.dwf.FDwfDigitalOutDataInfo(device_data.handle, ctypes.c_int(0), ctypes.byref(temp1)) == 0:
            self.check_error()
        device_data.digital.output.max_buffer_size = temp1.value

        return device_data

    def close(self):
        """
            close a specific device
        """
        if self.dwf.FDwfDigitalUartReset(self.dev.handle) == 0:
            self.check_error()
            
        if self.dev.handle != 0:
            self.dwf.FDwfDeviceClose(self.dev.handle)
        data.handle = ctypes.c_int(0)
        data.name = ""
        return