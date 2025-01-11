import struct

def parse_seizure_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    # Step 1: Extract header (ASCII until first null byte '\x00')
    header_end = content.find(b'\x00')
    header = content[:header_end].decode('ascii', errors='ignore')
    print("Header:", header)

    # Step 2: Extract binary data after the header
    binary_data = content[header_end + 1:]
    print(f"Binary data length: {len(binary_data)} bytes")

    # Step 3: Parse binary data dynamically
    seizure_times = []
    record_size = 8  # Expected record size: 4 bytes for start + 4 bytes for end
    for i in range(0, len(binary_data), record_size):
        if i + record_size <= len(binary_data):  # Ensure enough bytes remain for unpacking
            start_time, end_time = struct.unpack('<ii', binary_data[i:i + record_size])  # Little-endian integers
            seizure_times.append((start_time, end_time))
        else:
            print(f"Skipped incomplete record at offset {i}")

    return header, seizure_times

# Parse and print the file content
file_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\physionet.org\\files\\chbmit\\1.0.0\\chb01\\chb01_03.edf.seizures"
header, seizure_times = parse_seizure_file(file_path)
print("Header Info:", header)
print("Seizure Annotations (Start, End):", seizure_times)
