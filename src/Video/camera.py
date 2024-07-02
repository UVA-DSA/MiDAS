import os
from typing import Tuple, List
from abc import ABC, abstractmethod

from multiprocessing import Queue
from time import time_ns
import csv

import pyrealsense2 as rs
import numpy as np
import cv2

import queue
from PIL import ImageTk,Image


class Camera3D(ABC):
    def __init__(self, buffer_queue: Queue, root_dir: str, *args, **kwargs) -> None:
        self._queue = buffer_queue
        self._root_dir = root_dir
        self.save_path = os.path.join(self._root_dir, "camera")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.files_path = os.path.join(self.save_path, "frames")
        if not os.path.exists(self.files_path):
            os.makedirs(self.files_path)

        # Create a CSV file and write the header row
        csv_path = os.path.join(self.save_path, 'timestamps.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time','ID'])

        self._init_camera(*args, **kwargs)


    def _add_record(self, timestamp, id):
        self.csv_writer.writerow([timestamp, id])


    def _init_camera(self, *args, **kwargs) -> None:
        '''
        implement the necessary camera instantiations
        '''
        pass

    @abstractmethod
    def _get_image_depth_PCL(self) -> Tuple:
        '''
        retrieve the RGB, Depth and Point Cloud data from the camera
        must be individually implemented for any camera
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def run(self) -> None:
        '''
        this method is used to implement the worker thread.
        must read camera images whenever available, and write the data to the internal queue
        '''
        raise NotImplementedError()
        
class IntelCamera(Camera3D):

    # configs for the intel 3D camera capture
    RGB_DIM = (640, 480) # dimension of the rgb frames
    RGB_FPS = 30 # frame-rate of the rgb stream
    DEPTH_DIM = (640, 480) # dimension of the depth frames
    DEPTH_FPS = 30 # frame-rate of the depth stream
    ALIGN_FRAMES = True # align the rgb image to the depth image
    USE_DECIMATION_FILTER = False
    USE_SPATIAL_FILTER = True
    USE_TEMPORAL_FILTER = True
    USE_HOLE_FILLING_FILTER = True

    def __init__(self, buffer_queue: Queue, root_dir: str, *args, **kwargs) -> None:
        super().__init__(buffer_queue, root_dir, *args, **kwargs)

    def _init_camera(self, *args, **kwargs) -> None:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print(f"Camera device: {device_product_line} is connected")

        config.enable_stream(rs.stream.depth, *IntelCamera.DEPTH_DIM, rs.format.z16, IntelCamera.DEPTH_FPS)
        config.enable_stream(rs.stream.color, *IntelCamera.RGB_DIM, rs.format.bgr8, IntelCamera.RGB_FPS)
        config.enable_record_to_file(os.path.join(self.files_path, 'frames.bag'))

        # Start streaming
        self.profile = self.pipeline.start(config)

    def _get_image_depth_PCL(self) -> Tuple:
        pass

    def _save_frames(self):
        pass

    def _get_filters(self) -> List:
        # the set of filters to be applied to the frames
        _filters = []

        # Decimation filter
        if IntelCamera.USE_DECIMATION_FILTER:
            decimation = rs.decimation_filter()
            _filters.append(("decimation", decimation))

        # Spatial filter
        if IntelCamera.USE_SPATIAL_FILTER:
            spatial = rs.spatial_filter()
            _filters.append(("spatial", spatial))

        # Temporal filter
        if IntelCamera.USE_TEMPORAL_FILTER:
            temporal = rs.temporal_filter()
            _filters.append(("temporal", temporal))

        # Hole filling filter
        if IntelCamera.USE_HOLE_FILLING_FILTER:
            hole_filling = rs.hole_filling_filter()
            _filters.append(("hole_filling", hole_filling))

        return _filters
    
    def _apply_filters(self, frame, _filters):

        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        filters = {x[0]: x[1] for x in _filters}
        if "decimation" in filters:
            frame = filters["decimation"].process(frame)

        if ("spatial" in filters):
            frame = depth_to_disparity.process(frame)
            frame = filters["spatial"].process(frame)
            frame = filters["temporal"].process(frame)
            frame = disparity_to_depth.process(frame)

        if "hole_filling" in filters:
            frame = filters["hole_filling"].process(frame)

        return frame


    def run(self) -> None:
        colorizer = rs.colorizer()
        try:
            
            # build the post-processing depth filters
            _filters = self._get_filters()
            
            # if distance-based filtering is required. distance_in_meters = scale * depth_map
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

            # indexing the captured frames
            frame_num = -1

            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                frame_num += 1
                timestamp = time_ns()

                # align filter to align the depth frame to the color frame
                if IntelCamera.ALIGN_FRAMES:
                    align_to = rs.stream.color
                    align = rs.align(align_to)
                    frames = align.process(frames)

                # get the frames
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print(f"Error: a frame was missing")
                    continue

                # apply depth filters
                depth_frame = self._apply_filters(depth_frame, _filters)
                depth_frame = colorizer.process(depth_frame)

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_colormap = depth_image

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))


                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    self.finish()
                    break

                self._add_record(timestamp, frame_num)
                self._queue.put([timestamp, frame_num])


        finally:

            # Stop streaming
            self.pipeline.stop()

    def _dump_settings(self):
        '''
        Dump the settings of the camera into a JSON file for future reference
        '''
        params = dict()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        params['depth_scale'] = depth_scale
    
    def finish(self):
        self.pipeline.stop()
        self.csv_file.close()



class ZedCamera(Camera3D):
    def __init__(self, buffer_queue: Queue, root_dir: str, *args, **kwargs) -> None:
        super().__init__(buffer_queue, root_dir, *args, **kwargs)

    def _init_camera(self, *args, **kwargs) -> None:
        pass

    def _get_image_depth_PCL(self) -> Tuple:
        pass

    def run(self) -> None:
        while True:
            pass


class SimpleCamera(Camera3D):
    def __init__(self, buffer_queue: Queue, root_dir: str, *args, **kwargs) -> None:
        super().__init__(buffer_queue, root_dir, *args, **kwargs)

def _add_record(csv_writer, timestamp, id):
    csv_writer.writerow([timestamp, id])


def _apply_filters(frame, _filters):

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filters = {x[0]: x[1] for x in _filters}
    if "decimation" in filters:
        frame = filters["decimation"].process(frame)

    if ("spatial" in filters):
        frame = depth_to_disparity.process(frame)
        frame = filters["spatial"].process(frame)
        frame = filters["temporal"].process(frame)
        frame = disparity_to_depth.process(frame)

    if "hole_filling" in filters:
        frame = filters["hole_filling"].process(frame)

    return frame

def _get_filters() -> List:
    # the set of filters to be applied to the frames
    _filters = []

    # Decimation filter
    if IntelCamera.USE_DECIMATION_FILTER:
        decimation = rs.decimation_filter()
        _filters.append(("decimation", decimation))

    # Spatial filter
    if IntelCamera.USE_SPATIAL_FILTER:
        spatial = rs.spatial_filter()
        _filters.append(("spatial", spatial))

    # Temporal filter
    if IntelCamera.USE_TEMPORAL_FILTER:
        temporal = rs.temporal_filter()
        _filters.append(("temporal", temporal))

    # Hole filling filter
    if IntelCamera.USE_HOLE_FILLING_FILTER:
        hole_filling = rs.hole_filling_filter()
        _filters.append(("hole_filling", hole_filling))

    return _filters

def intel_camera_handler(q: Queue, path: str, img_q: Queue):
    _root_dir = path
    save_path = os.path.join(_root_dir, "camera")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_path = os.path.join(save_path, "frames")
    if not os.path.exists(files_path):
        os.makedirs(files_path)

    # Create a CSV file and write the header row
    csv_path = os.path.join(save_path, 'timestamps.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time','ID'])

    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(f"Camera device: {device_product_line} is connected")

    config.enable_stream(rs.stream.depth, *IntelCamera.DEPTH_DIM, rs.format.z16, IntelCamera.DEPTH_FPS)
    config.enable_stream(rs.stream.color, *IntelCamera.RGB_DIM, rs.format.bgr8, IntelCamera.RGB_FPS)
    config.enable_record_to_file(os.path.join(files_path, 'frames.bag'))

    # Start streaming
    profile = pipeline.start(config)

    colorizer = rs.colorizer()
    try:
            
        # build the post-processing depth filters
        _filters = _get_filters()
        
        # if distance-based filtering is required. distance_in_meters = scale * depth_map
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # indexing the captured frames
        frame_num = -1

        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frame_num += 1
            timestamp = time_ns()

            # align filter to align the depth frame to the color frame
            if IntelCamera.ALIGN_FRAMES:
                align_to = rs.stream.color
                align = rs.align(align_to)
                frames = align.process(frames)

            # get the frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                print(f"Error: a frame was missing")
                continue

            # apply depth filters
            depth_frame = _apply_filters(depth_frame, _filters)
            depth_frame = colorizer.process(depth_frame)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap = depth_image

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)

            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break

            # image compression and conversion to be sent to display
            img_q.put(images)

            _add_record(csv_writer, timestamp, frame_num)
            
            try:
                q.put([timestamp, frame_num], block=False)
            except:
                print("Error when writing to the FIFO: Clearing the queue ")
                while not q.empty():
                    q.get() # clear the queue


    finally:

        # Stop streaming
        pipeline.stop()

def zed_camera_handler(q: Queue, path: str):
    pass

def simple_camera_handler(q: Queue, path: str):
    pass

def get_camera_handler(cam_type: str = "Intel"):
    if cam_type == "Intel":
        return intel_camera_handler
    elif cam_type == "Zed":
        return zed_camera_handler
    else:
        raise ValueError(f"Camera type is not supported. Choose from [Intel, Zed]")
    
if __name__ == "__main__":
    test = Queue()
    path = "/test"

    intel_camera_handler(test, path)
    