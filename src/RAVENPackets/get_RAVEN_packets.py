import threading
import time
from pyge.emulators import PacketLoggerEmulator
import socket
import struct
import lz4
from collections import namedtuple
from config import LOGGER_PORT, RECEIVER_PORT

fields = 'sequence pactyp version delx0 delx1 dely0 dely1 delz0 delz1 Qx0 Qx1 Qy0 Qy1 Qz0 Qz1 Qw0 Qw1 buttonstate0 buttonstate1 grasp0 grasp1 surgeon_mode checksum'.split()
UStruct = namedtuple('UStruct', fields)
format_str = '<IIIiiiiiiddddddddiiiiii'

def analyze_log(log_file: str) -> dict:
    """
    Analyze the packet log to verify all packets were properly logged.
    
    Args:
        log_file: Path to the log file
    
    Returns:
        dict: Statistics about the logged packets
    """
    stats = {
        'total_packets': 0,
        'unique_packets': set(),
        'duplicate_packets': [],
        'missing_packets': [],
        'out_of_order': 0,
        'first_seq': None,
        'last_seq': None
    }
    
    print(f"[Analyzer] Reading log file: {log_file}")
    
    try:
        with lz4.frame.open(log_file, 'rb') as f:
            last_seq = -1
            while True:
                # Read packet header
                header = f.read(11)  # 8 (timestamp) + 2 (length) + 1 (dropped flag)
                if not header:
                    break
                    
                ts, length, dropped = struct.unpack("!QH?", header)
                data = f.read(length)
                
                unpacked_data = struct.unpack(format_str, data)
                unpacked_dict = UStruct(*unpacked_data)._asdict()
                print(unpacked_dict)
            
    except FileNotFoundError:
        print(f"[Error] Log file not found: {log_file}")
        return stats
    except Exception as e:
        print(f"[Error] Failed to analyze log: {str(e)}")
        return stats
    return stats

def start_logger(q, path, thread_stop):
    # Start Packet Logger
    print("[RAVEN] Starting packet logger...")
    logger = PacketLoggerEmulator(
        input_port=LOGGER_PORT,
        output_port=RECEIVER_PORT,
        protocol='udp',
        log_path=f"{path}/RAVEN_packets.bin"
    )
    # Start receiver first
    print("[RAVEN] Starting receiver thread...")
    
    # Start logger
    print("[RAVEN] Starting logger...")
    logger.start()
    while(not thread_stop.is_set()):
        continue

    # Stop logger
    print("[RAVEN] Stopping logger...")
    logger.stop()
    print("[RAVEN] Logger stopped")

if __name__ == "__main__":
    # Configuration
    LOGGER_PORT = 36000
    RECEIVER_PORT = 5000
    LOG_FILE = "RAVEN_packets.bin"
    
    # Start Packet Logger
    print("[Main] Starting packet logger...")
    logger = PacketLoggerEmulator(
        input_port=LOGGER_PORT,
        output_port=RECEIVER_PORT,
        protocol='udp',
        log_path=LOG_FILE
    )

   
    # Start receiver first
    print("[Main] Starting receiver thread...")
    
    # Start logger
    print("[Main] Starting logger...")
    logger.start()
    time.sleep(10)  # Wait for logger to initialize
        
    # Stop logger
    print("[Main] Stopping logger...")
    logger.stop()
    print("[Main] Logger stopped")

    # Analyze the log file
    print("\n[Main] Analyzing log file...")
    stats = analyze_log(LOG_FILE)