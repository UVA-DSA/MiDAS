

import cv2
import pytesseract
from PIL import Image
import os
import csv

# install tesseract from here: https://github.com/UB-Mannheim/tesseract/wiki
# install in the defualt path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def visualize_regions_on_frame(frame, regions):
    """
    Visualizes the specified regions on a video frame by drawing rectangles around them.

    :param frame: The frame (image) from the video.
    :param regions: List of regions to visualize. Each region is a tuple (x, y, width, height).
    """
    # Draw rectangles around the specified regions
    for i, region in enumerate(regions):
        x, y, w, h = region
        # Draw the rectangle (blue color, thickness 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Optionally, add a label with the region number
        label = f"Region {i+1}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with the visualized regions
    cv2.imwrite('frame.png', frame)
    
    # Wait indefinitely until a key is pressed, then close the display window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def extract_text_from_video_to_csv(video_path, regions, output_csv="output.csv"):
    """
    Extracts text from specific regions of each frame in a video and saves the results in a CSV file.

    :param video_path: Path to the .mkv video file.
    :param regions: List of regions to extract text from. Each region is a tuple (x, y, width, height).
    :param output_csv: Path to the output CSV file.
    """
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        # Initialize CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        headers = ["Frame ID"] + ["Instrument_1", "Instrument_3", "Instrument_4"]
        csv_writer.writerow(headers)

        # Capture the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every_x_seconds = 10


        frame_number = 0
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            row = [frame_number]
            if frame_number % (fps * every_x_seconds) == 0:
                print(frame_number, " of ", total_frames, " is being processed")
                for i, region in enumerate(regions):
                    x, y, w, h = region
                    # Crop the region from the frame
                    cropped_image = frame[y:y+h, x:x+w]
                    
                    # Convert the cropped region to a PIL image
                    pil_image = Image.fromarray(cropped_image)
                    
                    # Use Tesseract to extract text from the cropped image
                    text = pytesseract.image_to_string(pil_image).strip()
                    
                    row.append(text)
                
                # Write the row to the CSV file
                csv_writer.writerow(row)

            frame_number += 1

        # Release the video capture object
        cap.release()

if __name__ == "__main__":
    # Example usage:
    video_path = "D:/Data MIDAS/Bowel_S216_T1_2024-07-18/obs/2024-07-18 15-48-29.mkv"
    regions = [
        (380, 998, 180, 1042-998),  # (x, y, width, height)
        (1016, 998, 180, 45),
        (1332, 998, 180, 45),
        # Add more regions as needed
    ]
    # extract_text_from_video_to_csv(video_path, regions)

    # Region Test
    cap = cv2.VideoCapture(video_path)
    for i in range(300):
        cap.read()
    ret, frame = cap.read()
    visualize_regions_on_frame(frame, regions=regions)
