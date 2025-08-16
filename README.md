# Crowd_Detection
Final projects using YOLO, and O-openCV


## Overview
This project analyzes crowd density in video streams using:
- **YOLOv5** for human detection
- **DeepSORT** for object tracking
- **OpenCV** for video processing

The system:
1. Detects people in video frames
2. Tracks individuals across frames
3. Calculates crowd density in real-time
4. Classifies scenes as "RAMAI" (crowded) or "SEPI" (sparse)
5. Generates annotated output videos

## Setup Instructions

### Prerequisites
- Python 3.7+
- NVIDIA GPU (recommended)
- Google Colab (for cloud execution)

### Installation
```bash
!pip install opencv-python-headless numpy torch torchvision
!pip install yolov5
!pip install deep-sort-realtime
```

## Key Components

### 1. Model Initialization
```python
# Set computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize YOLOv5 (human detection)
model = YOLOv5("yolov5s.pt", device=device)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)
```

### 2. Detection and Tracking Workflow
```python
# Process each video frame
while cap.isOpened():
    ret, frame = cap.read()
    
    # Detect people using YOLOv5
    results = model.predict(frame)
    detections = results.pred[0][results.pred[0][:, 5] == 0].cpu().numpy()
    
    # Format detections for DeepSORT
    formatted_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        w, h = x2 - x1, y2 - y1
        formatted_detections.append(([x1, y1, w, h], conf, int(cls)))
    
    # Update tracker
    tracks = tracker.update_tracks(formatted_detections, frame=frame)
    
    # Process tracks and count people
    current_people = 0
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:
            current_people += 1
            # Draw bounding boxes and IDs
```

### 3. Density Classification
```python
# Set crowd threshold (adjustable)
crowd_threshold = 5

# Classify current frame
status = "RAMAI" if current_people >= crowd_threshold else "SEPI"

# Update historical data
status_history.append(status)
```

## Output Analysis
The system provides:
- Annotated video with bounding boxes and IDs
- Real-time people count display
- Scene classification ("RAMAI"/"SEPI")
- Final report including:
  - Maximum people detected
  - Overall scene classification
  - Frame-by-frame statistics

Sample output:
```
Analysis Complete!
Maximum people detected in a single frame: 12
Overall area status: RAMAI
RAMAI frames: 142, SEPI frames: 58
```

## Usage Guide
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Specify input video path:
```python
video_path = "/content/drive/MyDrive/path/to/your/video.mp4"
```

3. Run analysis:
```python
output_video, max_people, overall_status = analyze_crowd_density(
    video_path, 
    crowd_threshold=5
)
```

4. Download results:
```python
from google.colab import files
files.download(output_video)
```

## Performance Considerations
- **Hardware Acceleration:** Uses CUDA when available
- **Tracking Optimization:** 
  - `max_age=30` balances track persistence and false positive removal
  - Confidence threshold filtering reduces false detections
- **Memory Management:** 
  - Processes frames sequentially
  - Releases resources after completion

## Customization Options
1. Adjust crowd threshold:
```python
analyze_crowd_density(video_path, crowd_threshold=10)
```

2. Modify visualization:
```python
# Change bounding box color (BGR format)
cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue boxes

# Change text properties
cv2.putText(frame, ... , fontScale=0.8, color=(0,255,0))
```

3. Use different YOLOv5 models:
```python
# Larger model (more accurate, slower)
model = YOLOv5("yolov5l.pt", device=device)

# Smaller model (faster, less accurate)
model = YOLOv5("yolov5n.pt", device=device)
```

## Limitations and Future Improvements
**Current Limitations:**
- Occlusion handling in dense crowds
- Perspective distortion in wide-angle shots
- Lighting/shadows affecting detection

**Improvement Opportunities:**
1. Implement perspective normalization
2. Add people counting in specific zones
3. Integrate density heatmap visualization
4. Add crowd velocity analysis
5. Implement fall detection capabilities

## Sebelum Output
https://drive.google.com/file/d/1NBQv7p-G04rqLFbh2wpITsoYty-z_ZF3/view?usp=drive_link

## Setelah Output
https://drive.google.com/file/d/1Kyp-KBaIYJOurbWBa1QREFbyru3YRXa9/view?usp=drive_link
