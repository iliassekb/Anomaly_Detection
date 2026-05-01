"""Run this once to find the index of your phone virtual camera (iVCam / iriun)."""
import cv2

print("Scanning camera indices 0-9 …\n")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"  [{i}] ✓  {w}x{h}")
        cap.release()
    else:
        print(f"  [{i}] —  not available")

print("\nUse the index marked ✓ that matches your phone camera as SOURCE.")
