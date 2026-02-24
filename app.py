import os
import cv2
import time
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv

# We will use the local inference engine
try:
    from inference.models.utils import get_roboflow_model
    from inference.core.utils.image_utils import load_image_rgb
    HAS_INFERENCE_SDK = True
except ImportError:
    HAS_INFERENCE_SDK = False

load_dotenv()

app = Flask(__name__)

# Replace with your actual model details and API key from Roboflow
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "shoplifting-detection/1")

# Get camera source from .env, default to 0 (webcam)
cam_src = os.environ.get("CAMERA_SOURCE", "0")
if cam_src.isdigit():
    CAMERA_SOURCE = int(cam_src)
else:
    CAMERA_SOURCE = cam_src

# Initialize the local Inference Model if available
local_model = None
if HAS_INFERENCE_SDK and ROBOFLOW_API_KEY:
    try:
         # get_roboflow_model will download the model weights ONCE to the local machine
         # and then run entirely locally using CPU or GPU.
         local_model = get_roboflow_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
         print(f"Successfully loaded local model: {MODEL_ID}")
    except Exception as e:
        print(f"Error initializing local model: {e}")

# This will hold the latest detection results to show on the UI
latest_detections = []
latest_snapshot = None  # URL/path to the most recent snapshot

# Global config arrays
CONFIDENCE_THRESHOLD = 0.40
PROCESSING_FRAME_SKIP = 3

# Ensure snapshot directory exists
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'snapshots')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def generate_frames():
    global latest_detections, CAMERA_SOURCE, latest_snapshot, CONFIDENCE_THRESHOLD, PROCESSING_FRAME_SKIP
    
    last_snapshot_time = 0
    frame_count = 0
    
    current_source = CAMERA_SOURCE
    cap = cv2.VideoCapture(current_source) 

    if not cap.isOpened():
        print(f"Error: Could not open camera source: {current_source}")
        # fallback to empty frame or break
        # we don't return here so it can keep retrying if the source changes via UI

    while True:
        # Check if the UI has updated the global target CAMERA_SOURCE
        if current_source != CAMERA_SOURCE:
            if cap and cap.isOpened():
                cap.release()
            current_source = CAMERA_SOURCE
            cap = cv2.VideoCapture(current_source)
            if not cap.isOpened():
                print(f"Error: Could not open NEW camera source: {current_source}")
                # Send a blank frame indicating stream is down, but keep loop alive
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                continue

        if cap and cap.isOpened():
            success, frame = cap.read()
        else:
            success = False

        if not success:
            # Send a blank frame or loop over to retry Source Update
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            continue
        
        frame_count += 1
        # We use the local model
        
        # Only run inference every N frames to save CPU/GPU, but draw last known boxes below
        if local_model and (frame_count % PROCESSING_FRAME_SKIP == 0):
             try:
                 # Local model expects RGB array or PIL image, OpenCV is BGR
                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 
                 # Run inference entirely on the local machine
                 results = local_model.infer(frame_rgb)
                 
                 # results is usually a list of ObjectDetectionPrediction objects or dicts depending on version
                 # Let's extract them cleanly
                 if isinstance(results, list) and len(results) > 0:
                     predictions = results[0].predictions # If batch size 1
                 else:
                     predictions = getattr(results, 'predictions', []) if not isinstance(results, dict) else results.get('predictions', [])
                     
                 new_detections = []
                 
                 # Process bounding boxes
                 for det in predictions:
                     # Handing different SDK return formats (dict or object)
                     x = det.x if hasattr(det, 'x') else det['x']
                     y = det.y if hasattr(det, 'y') else det['y']
                     w = det.width if hasattr(det, 'width') else det['width']
                     h = det.height if hasattr(det, 'height') else det['height']
                     label = det.class_name if hasattr(det, 'class_name') else det.get('class', 'unknown')
                     confidence = det.confidence if hasattr(det, 'confidence') else det['confidence']
                     
                     
                     # Simple logic: If we detect 'shoplifting' or a specific class
                     if confidence >= CONFIDENCE_THRESHOLD: # threshold
                         new_detections.append({
                             'label': label, 'confidence': confidence,
                             'x': x, 'y': y, 'w': w, 'h': h
                         })
                         
                 latest_detections = new_detections
                 
             except Exception as e:
                 print(f"Local Inference error: {e}")

        # Draw the latest known bounding boxes on every frame (persistence)
        for det in latest_detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            label = det['label']
            confidence = det['confidence']
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                 
        # Save a snapshot if there are detections (max 1 per 3 seconds)
        if len(latest_detections) > 0:
            current_time = time.time()
            if current_time - last_snapshot_time > 3.0:
                filename = f"snapshot_{int(current_time)}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                latest_snapshot = f"/static/snapshots/{filename}"
                last_snapshot_time = current_time
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Endpoint for the frontend to poll for the latest alerts/detections."""
    global CAMERA_SOURCE, latest_snapshot, CONFIDENCE_THRESHOLD, PROCESSING_FRAME_SKIP
    
    # Send a clean version of latest_detections without x,y,w,h coords to UI if desired, but fine either way
    ui_detections = [{'label': d['label'], 'confidence': d['confidence']} for d in latest_detections]
    
    return jsonify({
        "status": "active",
        "detections": ui_detections,
        "current_source": CAMERA_SOURCE,
        "snapshot_url": latest_snapshot,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "frame_skip": PROCESSING_FRAME_SKIP
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Endpoint to update the camera config dynamically from the UI."""
    global CAMERA_SOURCE, CONFIDENCE_THRESHOLD, PROCESSING_FRAME_SKIP
    data = request.json
    
    new_source = data.get('camera_source')
    new_conf = data.get('confidence_threshold')
    new_skip = data.get('frame_skip')
    
    if new_conf is not None:
        try:
            CONFIDENCE_THRESHOLD = float(new_conf)
        except ValueError:
            pass
            
    if new_skip is not None:
        try:
            PROCESSING_FRAME_SKIP = int(new_skip)
            if PROCESSING_FRAME_SKIP < 1:
                PROCESSING_FRAME_SKIP = 1
        except ValueError:
            pass
    
    if new_source is not None and new_source != "":
        # Convert to int if it's a digit (like "0" for webcam)
        if str(new_source).isdigit():
            CAMERA_SOURCE = int(new_source)
        else:
            CAMERA_SOURCE = str(new_source)
            
        return jsonify({"success": True, "message": f"Settings updated."})
    
    return jsonify({"success": True, "message": "Settings updated (no source change)."}), 200

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
