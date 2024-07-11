import os
from typing import Tuple, List
import time

from multiprocessing import Queue, Event
from time import time_ns
import csv

import pyrealsense2 as rs
import pyzed.sl as sl
import numpy as np
import cv2

import queue
from PIL import ImageTk,Image


class ReconnectException(Exception):
    "This exception is raised when camera is disconnected and a new connection is required"


def _add_record(csv_writer: csv.writer, timestamp, id):
    csv_writer.writerow([timestamp, id])

def _init_filesystem(path, cam_type: str):
    _root_dir = path
    save_path = os.path.join(_root_dir, "camera", cam_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_path = os.path.join(save_path, "frames")
    if not os.path.exists(files_path):
        os.makedirs(files_path)

    # Create a CSV file and write the header row
    csv_path = os.path.join(save_path, 'timestamps.csv')
    if not os.path.exists(csv_path):
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Time','ID'])
    else:
       csv_file = open(csv_path, 'a', newline='') 
       csv_writer = csv.writer(csv_file)

    return csv_writer, csv_file, files_path


################################################### Intel ################################################
     
class IntelCamera:

    # configs for the intel 3D camera capture
    RGB_DIM = (640, 480) # dimension of the rgb frames
    RGB_FPS = 30 # frame-rate of the rgb stream
    DEPTH_DIM = (640, 480) # dimension of the depth frames
    DEPTH_FPS = 30 # frame-rate of the depth stream
    ALIGN_FRAMES = True # align the rgb image to the depth image
    APPLY_FILTERS = False
    USE_DECIMATION_FILTER = False
    USE_SPATIAL_FILTER = True
    USE_TEMPORAL_FILTER = True
    USE_HOLE_FILLING_FILTER = True



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

def intel_cleanup(pipeline: rs.pipeline, csv_file):
    try:
        pipeline.stop()
        csv_file.close()
    except:
        pass

def intel_camera_handler(q: Queue, path: str, img_q: Queue, thread_stop):

    part_id = 0
    # build the post-processing depth filters
    _filters = _get_filters()

    while True:

        pipeline = rs.pipeline()
        config = rs.config()
        try:
            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            print(f"Camera device: {device_product_line} is connected")
        except:
            print("Intel Camera Not Found")
            return
        
        config.enable_stream(rs.stream.depth, *IntelCamera.DEPTH_DIM, rs.format.z16, IntelCamera.DEPTH_FPS)
        config.enable_stream(rs.stream.color, *IntelCamera.RGB_DIM, rs.format.bgr8, IntelCamera.RGB_FPS)

        if thread_stop.is_set():
            intel_cleanup(pipeline, csv_file)
            break

        csv_writer, csv_file, files_path = _init_filesystem(path, "Intel")
        config.enable_record_to_file(os.path.join(files_path, f'part{part_id}_frames.bag'))

        # Start streaming
        profile = pipeline.start(config)

        colorizer = rs.colorizer()
        try:
            
            # if distance-based filtering is required. distance_in_meters = scale * depth_map
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

            # indexing the captured frames
            frame_num = -1

            while True:

                if thread_stop.is_set():
                    intel_cleanup(pipeline, csv_file)
                    break

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
                if IntelCamera.APPLY_FILTERS:
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
                img_q.put(images, block=False)

                _add_record(csv_writer, timestamp, frame_num)
                
                q.put([timestamp, frame_num], block=False)


        except KeyboardInterrupt:
            # Stop streaming
            intel_cleanup(pipeline, csv_file)
            break

        except:
            print("HHHHHHHHHHHAAAAAAAAAAAAAAAAAA")
            # Stop streaming
            intel_cleanup(pipeline, csv_file)
            part_id += 1

    q.close()
    img_q.close()
    exit(0)


################################################### ZED ################################################
class ZedCamera:
    RESOLUTION = sl.RESOLUTION.HD1080
    FPS = 30
    EXPOSURE = -1 # % of framerate
    BRIGHTNESS = -1 # 0-8
    CONTRAST = -1 # 0-8
    DEPTH = sl.DEPTH_MODE.ULTRA # ULTRA, NEURAL,NEURAL_PLUS
    COMPRESSION = sl.SVO_COMPRESSION_MODE.H265
    FILL = False

def zed_cleanup(zed: sl.Camera, csv_file):
    zed.disable_recording()
    zed.close()
    csv_file.close()

def zed_camera_handler(q: Queue, path: str, img_q: Queue, thread_stop: Event):
    
    # configs
    init_params = sl.InitParameters()
    init_params.camera_resolution = ZedCamera.RESOLUTION
    init_params.camera_fps = ZedCamera.FPS
    init_params.depth_mode = ZedCamera.DEPTH
    init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)
    init_params.depth_minimum_distance = 120.0 # Set the minimum depth perception distance to 12cm
    
    
    recordingParameters = sl.RecordingParameters()
    recordingParameters.compression_mode = ZedCamera.COMPRESSION

    part_id = 0

    while True:

        if thread_stop.is_set():
            zed_cleanup(zed, csv_file)
            break

        csv_writer, csv_file, files_path = _init_filesystem(path, "Zed")
        recordingParameters.video_filename = os.path.join(files_path, f"part{part_id}_frames.svo")

        zed = sl.Camera()
        if ZedCamera.EXPOSURE != -1:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, ZedCamera.EXPOSURE)
        if ZedCamera.BRIGHTNESS != -1:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, ZedCamera.BRIGHTNESS)
        if ZedCamera.CONTRAST != -1:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, ZedCamera.CONTRAST)

        try:
            # Open the camera
            err = sl.ERROR_CODE.FAILURE
            while err != sl.ERROR_CODE.SUCCESS:
                err = zed.open(init_params)
                time.sleep(1)

            zed_serial = zed.get_camera_information().serial_number
            err = zed.enable_recording(recordingParameters)
            if err != sl.ERROR_CODE.SUCCESS:
                print("no recording initiated")

            frame_num = -1

            while True:
                if thread_stop.is_set():
                    zed_cleanup(zed, csv_file)
                    break
                image = sl.Mat()
                depth_map = sl.Mat()

                runtime_parameters = sl.RuntimeParameters()
                if ZedCamera.FILL:
                    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
                    # A new image and depth is available if grab() returns SUCCESS
                    zed.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE) # Retrieve left image
                    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
                    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured

                    img_q.put(image.get_data(), block=False)

                    frame_num += 1
                    _add_record(csv_writer, timestamp.get_nanoseconds(), frame_num)
                    
                    q.put([timestamp.get_nanoseconds(), frame_num], block=False)


                    # cv2.namedWindow('Zed', cv2.WINDOW_AUTOSIZE)
                    # cv2.imshow('Zed', image.get_data())

                    # key = cv2.waitKey(1)
                    # if key & 0xFF == ord('q') or key == 27:
                    #     cv2.destroyAllWindows()
                    #     raise Exception()
                else:
                    raise ReconnectException() # to trigger the exception handling and trying to reconnect


        except ReconnectException as e:
            print(e)
            part_id += 1
            zed_cleanup(zed, csv_file)

        except KeyboardInterrupt as e:
            print(e)
            zed_cleanup(zed, csv_file)
            break

    exit(0)



def get_camera_handler(cam_type: str = "Intel"):
    if cam_type == "Intel":
        return intel_camera_handler
    elif cam_type == "Zed":
        return zed_camera_handler
    else:
        raise ValueError(f"Camera type is not supported. Choose from [Intel, Zed]")
    
if __name__ == "__main__":
    cam_type = "Intel"
    q, iq = Queue(1000), Queue(1000)
    e = Event()
    path = "./test"

    handler = get_camera_handler(cam_type)
    handler(q, path, iq, e)
    e.clear()
    