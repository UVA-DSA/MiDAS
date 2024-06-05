
from multiprocessing import Queue
from typing import Tuple
from abc import ABC, abstractmethod

import pyrealsense2 as rs
import numpy as np
import cv2

class Camera3D(ABC):
    def __init__(self, buffer_queue: Queue, root_dir: str, *args, **kwargs) -> None:
        self._init_camera(*args, **kwargs)
        self._queue = buffer_queue
        self._root_dir = root_dir

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

        # Start streaming
        self.profile = self.pipeline.start(config)

    def _get_image_depth_PCL(self) -> Tuple:
        pass

    def run(self) -> None:
        try:
            _filters = []
            if IntelCamera.ALIGN_FRAMES:
                align_to = rs.stream.color
                align = rs.align(align_to)
                _filters.append(align)

            # if distance-based filtering is required. distance_in_meters = scale * depth_map
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                for f in _filters:
                    frames = f.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print(f"Error: a frame was missing")
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

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
                cv2.waitKey(1)

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

def get_camera_instance(cam_type: str = "Intel"):
    if cam_type == "Intel":
        return IntelCamera
    elif cam_type == "Zed":
        return ZedCamera
    else:
        raise ValueError(f"Camera type is not supported. Choose from [Intel, Zed]")
    
if __name__ == "__main__":
    cam = get_camera_instance("Intel")(Queue(), "")
    cam.run()