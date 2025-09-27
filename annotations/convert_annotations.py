#!/usr/bin/env python3
"""
Script to convert temporal video annotations to frame-by-frame gesture labels.

This script reads a VGG Video Annotator CSV file with temporal annotations
and generates a new CSV file where each frame has its corresponding gesture label.
"""

import csv
import json
import re
from typing import Dict, List, Tuple, Optional
import argparse


class TemporalAnnotationProcessor:
    """Process temporal video annotations and convert to frame-by-frame labels."""
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize the processor.
        
        Args:
            fps: Frames per second of the video (default: 30.0)
        """
        self.fps = fps
        
        # Gesture ID to name mapping from the requirements
        self.gesture_mapping = {
            'S1': 'Approach peg',
            'S2': 'Align & grasp',
            'S3': 'Lift peg',
            'S4': 'Transfer peg - Get together',
            'S5': 'Transfer peg - Exchange',
            'S6': 'Approach pole',
            'S7': 'Align & place',
            'Idle': 'Idle'
        }
    
    def parse_annotation_file(self, filepath: str) -> Tuple[Dict[str, str], List[Dict]]:
        """
        Parse the VGG Video Annotator CSV file.
        
        Args:
            filepath: Path to the annotation CSV file
            
        Returns:
            Tuple of (gesture_options_dict, list_of_annotations)
        """
        gesture_options = {}
        annotations = []
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Parse header lines (0-9) to extract gesture options
        for i, line in enumerate(lines[:10]):
            if 'ATTRIBUTE' in line and 'Gesture' in line:
                # Extract gesture options from the attribute line
                # Line format: # ATTRIBUTE = {"1":{...},"2":{"aname":"Gesture",...,"options":{"0":"S1","1":"S2",...},...}}
                match = re.search(r'"options":\{([^}]+)\}', line)
                if match:
                    options_str = '{' + match.group(1) + '}'
                    try:
                        options_dict = json.loads(options_str)
                        gesture_options = options_dict
                        print(f"Found gesture options: {gesture_options}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing gesture options: {e}")
        
        # Parse data lines (starting from line 10)
        csv_reader = csv.reader(lines[10:])
        for row in csv_reader:
            if len(row) >= 6:  # Ensure we have all required columns
                try:
                    # Parse temporal coordinates
                    temporal_coords_str = row[3]  # temporal_coordinates column
                    temporal_coords = json.loads(temporal_coords_str)
                    
                    # Parse metadata to get gesture ID
                    metadata_str = row[5]  # metadata column
                    metadata = json.loads(metadata_str)
                    
                    # Extract gesture ID (attribute "2" is the gesture)
                    gesture_id = metadata.get("2", "7")  # Default to "7" (Idle) if not found
                    
                    annotation = {
                        'start_time': float(temporal_coords[0]),
                        'end_time': float(temporal_coords[1]) if len(temporal_coords) > 1 else float(temporal_coords[0]),
                        'gesture_id': gesture_id,
                        'gesture_code': gesture_options.get(gesture_id, 'Idle'),
                        'gesture_name': self.gesture_mapping.get(gesture_options.get(gesture_id, 'Idle'), 'Idle')
                    }
                    annotations.append(annotation)
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error parsing annotation row: {row}")
                    print(f"Error details: {e}")
                    continue
        
        return gesture_options, annotations
    
    def get_video_duration(self, annotations: List[Dict]) -> float:
        """
        Calculate the total video duration from annotations.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Total video duration in seconds
        """
        if not annotations:
            return 0.0
        
        max_time = max(ann['end_time'] for ann in annotations)
        return max_time
    
    def generate_frame_labels(self, annotations: List[Dict], video_duration: float) -> List[Dict]:
        """
        Generate frame-by-frame gesture labels from temporal annotations.
        
        Args:
            annotations: List of temporal annotations
            video_duration: Total video duration in seconds
            
        Returns:
            List of frame dictionaries with labels
        """
        # Calculate total number of frames
        total_frames = int(video_duration * self.fps) + 1
        
        # Initialize all frames with default label
        frame_labels = []
        for frame_num in range(total_frames):
            timestamp = frame_num / self.fps
            frame_labels.append({
                'frame_number': frame_num,
                'timestamp': timestamp,
                'gesture_code': 'Idle',
                'gesture_name': 'Idle'
            })
        
        # Sort annotations by start time to handle overlaps properly
        sorted_annotations = sorted(annotations, key=lambda x: x['start_time'])
        
        # Apply annotations to frames
        for annotation in sorted_annotations:
            start_frame = int(annotation['start_time'] * self.fps)
            end_frame = int(annotation['end_time'] * self.fps)
            
            # Ensure we don't exceed frame bounds
            start_frame = max(0, start_frame)
            end_frame = min(total_frames - 1, end_frame)
            
            # Apply gesture label to all frames in the temporal segment
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(frame_labels):
                    frame_labels[frame_idx]['gesture_code'] = annotation['gesture_code']
                    frame_labels[frame_idx]['gesture_name'] = annotation['gesture_name']
        
        return frame_labels
    
    def save_frame_labels_csv(self, frame_labels: List[Dict], output_filepath: str):
        """
        Save frame-by-frame labels to a CSV file.
        
        Args:
            frame_labels: List of frame label dictionaries
            output_filepath: Path to save the output CSV file
        """
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['frame_number', 'timestamp', 'gesture_code', 'gesture_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write frame data
            for frame_data in frame_labels:
                writer.writerow(frame_data)
        
        print(f"Frame-by-frame labels saved to: {output_filepath}")
        print(f"Total frames processed: {len(frame_labels)}")
    
    def process_annotations(self, input_filepath: str, output_filepath: str) -> Dict:
        """
        Main processing function to convert temporal annotations to frame labels.
        
        Args:
            input_filepath: Path to input annotation CSV file
            output_filepath: Path to output frame labels CSV file
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"Processing annotations from: {input_filepath}")
        print(f"Video FPS: {self.fps}")
        
        # Parse the annotation file
        gesture_options, annotations = self.parse_annotation_file(input_filepath)
        
        if not annotations:
            raise ValueError("No valid annotations found in the file")
        
        print(f"Found {len(annotations)} temporal annotations")
        
        # Calculate video duration
        video_duration = self.get_video_duration(annotations)
        print(f"Video duration: {video_duration:.2f} seconds")
        
        # Generate frame-by-frame labels
        frame_labels = self.generate_frame_labels(annotations, video_duration)
        
        # Save to CSV
        self.save_frame_labels_csv(frame_labels, output_filepath)
        
        # Generate statistics
        gesture_counts = {}
        for frame in frame_labels:
            gesture = frame['gesture_name']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        stats = {
            'total_annotations': len(annotations),
            'total_frames': len(frame_labels),
            'video_duration': video_duration,
            'fps': self.fps,
            'gesture_distribution': gesture_counts
        }
        
        return stats


def main():
    """Main function to run the annotation processor."""
    parser = argparse.ArgumentParser(description='Convert temporal video annotations to frame-by-frame labels')
    parser.add_argument('input_file', help='Input CSV annotation file')
    parser.add_argument('-o', '--output', default=None, help='Output CSV file (default: input_file_frames.csv)')
    parser.add_argument('--fps', type=float, default=30.0, help='Video frames per second (default: 30.0)')
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output is None:
        input_name = args.input_file.rsplit('.', 1)[0]
        output_file = f"{input_name}_frames.csv"
    else:
        output_file = args.output
    
    try:
        # Create processor and process annotations
        processor = TemporalAnnotationProcessor(fps=args.fps)
        stats = processor.process_annotations(args.input_file, output_file)
        
        # Print statistics
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total temporal annotations: {stats['total_annotations']}")
        print(f"Total frames generated: {stats['total_frames']}")
        print(f"Video duration: {stats['video_duration']:.2f} seconds")
        print(f"FPS: {stats['fps']}")
        print("\nGesture distribution:")
        for gesture, count in sorted(stats['gesture_distribution'].items()):
            percentage = (count / stats['total_frames']) * 100
            print(f"  {gesture}: {count} frames ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error processing annotations: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
