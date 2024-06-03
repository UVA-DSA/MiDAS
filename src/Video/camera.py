class Camera3D:
    def __init__(self, buffer_queue, *args, **kwargs) -> None:
        self._init_camera(*args, **kwargs)
        self.__queue = buffer_queue

    def _init_camera(self, *args, **kwargs):
        '''
        implement the necessary camera instantiations
        '''
        pass

    def _get_image_depth_PCL(self):
        '''
        retrieve the RGB, Depth and Point Cloud data from the camera
        must be individually implemented for any camera
        '''
        raise NotImplementedError()
    
    def run(slef):
        '''
        this method is used to implement the worker thread.
        must read camera images whenever available, and write the data to the internal queue
        '''
        raise NotImplementedError()


class ZedCamera(Camera3D):
    pass

class IntelCamera(Camera3D):
    pass

class SimpleCamera(Camera3D):
    pass