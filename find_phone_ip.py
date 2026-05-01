"""Scan local network and list all active devices with open port 8080."""
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def check(ip):
    try:
        with socket.create_connection((ip, 8080), timeout=0.5):
            return ip
    except OSError:
        return None

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
local_ip = s.getsockname()[0]
s.close()

prefix = ".".join(local_ip.split(".")[:3])
print(f"Your PC IP : {local_ip}")
print(f"Scanning   : {prefix}.1-254 on port 8080 …\n")

found = []
with ThreadPoolExecutor(max_workers=80) as pool:
    futures = {pool.submit(check, f"{prefix}.{i}"): i for i in range(1, 255)}
    for fut in as_completed(futures):
        ip = fut.result()
        if ip:
            found.append(ip)

if found:
    print("Devices found with port 8080 open:")
    for ip in sorted(found):
        print(f"  → {ip}:8080   (use this IP in the script)")
else:
    print("No device found on port 8080.")
    print("Make sure IP Webcam is running and 'Start server' is tapped.")
