import cv2
import numpy as np
import csv

# Function to detect the pattern in the specified region
def detect_pattern_in_region(frame, pattern, region):
    x, y, w, h = region
    region_of_interest = frame[y:y+h, x:x+w]
    result = cv2.matchTemplate(region_of_interest, pattern, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # You can adjust the threshold as needed
    loc = np.where(result >= threshold)
    return len(loc[0]) > 0  # Returns True if pattern is detected

# Main function to process the video and detect the pattern
def detect_pattern_in_video(video_path, pattern_path, regions, output_csv):
    # Load the pattern image
    pattern = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_id = -1
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"Processing frame:{frame_id} / {total_frames}")
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for pattern in each region
        detection_results = []
        for region in regions:
            detected = detect_pattern_in_region(gray_frame, pattern, region)
            detection_results.append(detected)
        
        # Add the results to the list
        results.append([frame_id] + detection_results)

    cap.release()
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Frame_ID', 'Instrument_1', 'Instrument_2', 'Instrument_3']
        csvwriter.writerow(header)
        csvwriter.writerows(results)

# Example usage
video_path = '2024-07-18 15-41-56.mkv'
pattern_path = 'pattern.png'
h = w = 11
regions = [
    (365, 999, w, h),   # Instrument 1
    (999, 999, w, h),   # Instrument 2
    (1316, 999, w, h)   # Instrument 3
]
output_csv = 'clutch_events.csv'

detect_pattern_in_video(video_path, pattern_path, regions, output_csv)
