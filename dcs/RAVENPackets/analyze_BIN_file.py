import threading
import time
from pyge.emulators import PacketLoggerEmulator
import socket
import struct
import lz4
from collections import namedtuple

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
                unpacked_dict["timestamp"] = ts
                print(unpacked_dict)
            
    except FileNotFoundError:
        print(f"[Error] Log file not found: {log_file}")
        return stats
    except Exception as e:
        print(f"[Error] Failed to analyze log: {str(e)}")
        return stats
    return stats

if __name__ == "__main__":
    analyze_log("./RAVEN_packets.bin")