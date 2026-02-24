import cv2

# The URL from the user
rtsp_url = "rtsp://192.168.178.22:554"

print(f"Testing RTSP URL: {rtsp_url}")
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("FAILED: Could not connect to the stream.")
else:
    print("SUCCESS: Connected to stream!")
    # Grab one frame just to be sure
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS: Read a frame of size {frame.shape}")
    else:
        print("FAILED: Connected but could not read a frame. Stream might be empty or encrypted.")

cap.release()
