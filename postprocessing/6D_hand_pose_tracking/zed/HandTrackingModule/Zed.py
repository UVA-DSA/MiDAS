"""Thin, configurable wrapper around the ZED SDK (``pyzed.sl``).

The wrapper serves two purposes:

* every capture setting (resolution, FPS, depth mode, depth range, units, ...)
  is a constructor argument instead of a hard-coded constant, so the same code
  runs on any ZED model and any recording;
* it papers over the API differences between ZED SDK 3.x, 4.x and 5.x
  (renamed enums, relocated calibration data, removed sensing modes) so a
  script written against one SDK generation keeps working on the others.

Typical use::

    from HandTrackingModule.Zed import Zed

    with Zed("recording.svo2", depth_mode="NEURAL") as cam:
        cam.print_information()
        while cam.grab():
            cam.get_image()
            ...
"""

import pyzed.sl as sl

# Defaults used by every script in this package. They are deliberately
# conservative: HD720@30 is supported by every ZED model.
DEFAULT_RESOLUTION = "HD720"
DEFAULT_FPS = 30
DEFAULT_DEPTH_MODE = "ULTRA"
DEFAULT_COORDINATE_UNITS = "METER"
DEFAULT_DEPTH_MIN = 0.3
DEFAULT_DEPTH_MAX = 40.0
DEFAULT_DEPTH_CONFIDENCE = 100

# A requested enum value may not exist on the installed SDK (for example
# DEPTH_MODE.ULTRA was removed in SDK 5.0). Each entry lists the preferred
# value first, followed by the closest available substitutes.
_DEPTH_MODE_FALLBACKS = {
    "NONE": ("NONE",),
    "PERFORMANCE": ("PERFORMANCE", "NEURAL_LIGHT", "QUALITY", "NEURAL"),
    "QUALITY": ("QUALITY", "ULTRA", "NEURAL", "PERFORMANCE"),
    "ULTRA": ("ULTRA", "NEURAL", "QUALITY", "PERFORMANCE"),
    "NEURAL_LIGHT": ("NEURAL_LIGHT", "NEURAL", "ULTRA", "PERFORMANCE"),
    "NEURAL": ("NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"),
    "NEURAL_PLUS": ("NEURAL_PLUS", "NEURAL", "ULTRA", "QUALITY"),
}

_RESOLUTION_FALLBACKS = {
    "AUTO": ("AUTO", "HD720"),
    "VGA": ("VGA", "SVGA", "HD720"),
    "SVGA": ("SVGA", "VGA", "HD720"),
    "HD720": ("HD720", "AUTO"),
    "HD1080": ("HD1080", "HD1200", "HD720"),
    "HD1200": ("HD1200", "HD1080", "HD720"),
    "HD2K": ("HD2K", "QHDPLUS", "HD1080"),
    "QHDPLUS": ("QHDPLUS", "HD2K", "HD1080"),
    "HD4K": ("HD4K", "HD2K", "HD1080"),
}

#: Names accepted by the ``--resolution`` / ``--depth-mode`` / ``--units``
#: command line options of the scripts shipped with this package.
RESOLUTION_CHOICES = tuple(_RESOLUTION_FALLBACKS)
DEPTH_MODE_CHOICES = tuple(_DEPTH_MODE_FALLBACKS)
COORDINATE_UNIT_CHOICES = ("MILLIMETER", "CENTIMETER", "METER", "INCH", "FOOT")


def resolve_enum(enum_cls, candidates, what):
    """Return ``(member, name)`` for the first member of `enum_cls` in `candidates`.

    Raises ``ValueError`` when none of the candidate names exist, which means
    the installed SDK is too old or too new for the requested value.
    """
    for name in candidates:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name), name
    raise ValueError(
        "None of {0} is available as a {1} in the installed ZED SDK.".format(
            ", ".join(candidates), what
        )
    )


def resolve_resolution(name):
    """Resolve a resolution name, substituting the closest available value."""
    key = str(name).upper()
    return resolve_enum(sl.RESOLUTION, _RESOLUTION_FALLBACKS.get(key, (key,)), "resolution")


def resolve_depth_mode(name):
    """Resolve a depth mode name, substituting the closest available value."""
    key = str(name).upper()
    return resolve_enum(sl.DEPTH_MODE, _DEPTH_MODE_FALLBACKS.get(key, (key,)), "depth mode")


class Zed(object):
    """Open a ZED camera live, or replay a recorded ``.svo`` / ``.svo2`` file.

    Args:
        filename: Path to an SVO recording. ``None`` opens the live camera.
        depth_confidence: Depth confidence threshold, 0-100. Lower values
            discard more uncertain depth pixels.
        resolution: One of :data:`RESOLUTION_CHOICES`. Ignored when replaying
            an SVO, whose resolution is fixed at recording time.
        fps: Requested capture frame rate for the live camera.
        depth_mode: One of :data:`DEPTH_MODE_CHOICES`. Neural modes are more
            accurate but need the AI models (see ``download_zed_models.py``).
        coordinate_units: Units of the returned 3D coordinates, one of
            :data:`COORDINATE_UNIT_CHOICES`.
        depth_minimum_distance / depth_maximum_distance: Depth clipping range,
            expressed in `coordinate_units`.
        svo_real_time_mode: ``True`` replays an SVO at its recorded frame rate
            (dropping frames to keep up). Leave ``False`` for batch processing
            so that every frame is delivered.
        verbose: Print the resolved settings while opening.
    """

    def __init__(
        self,
        filename=None,
        depth_confidence=DEFAULT_DEPTH_CONFIDENCE,
        resolution=DEFAULT_RESOLUTION,
        fps=DEFAULT_FPS,
        depth_mode=DEFAULT_DEPTH_MODE,
        coordinate_units=DEFAULT_COORDINATE_UNITS,
        depth_minimum_distance=DEFAULT_DEPTH_MIN,
        depth_maximum_distance=DEFAULT_DEPTH_MAX,
        svo_real_time_mode=False,
        verbose=True,
    ):
        self.verbose = verbose
        self._log("Bringing up ZED camera information...")

        # Decide if SVO or live
        self.input_type = sl.InputType()
        if filename is None:
            self._log("Using live stream from ZED camera")
            self.svo_mode = False
        else:
            self._log("Reading SVO file: {0}".format(filename))
            self.input_type.set_from_svo_file(str(filename))
            self.svo_mode = True
        self.filename = filename

        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)

        resolution_enum, resolved_resolution = resolve_resolution(resolution)
        depth_mode_enum, resolved_depth_mode = resolve_depth_mode(depth_mode)
        units_enum, _ = resolve_enum(
            sl.UNIT, (str(coordinate_units).upper(),), "coordinate unit"
        )

        if resolved_resolution != str(resolution).upper():
            self._log(
                "Resolution {0} unavailable in this SDK, using {1} instead.".format(
                    resolution, resolved_resolution
                )
            )
        if resolved_depth_mode != str(depth_mode).upper():
            self._log(
                "Depth mode {0} unavailable in this SDK, using {1} instead.".format(
                    depth_mode, resolved_depth_mode
                )
            )

        self.init_params.camera_resolution = resolution_enum
        self.init_params.camera_fps = int(fps)
        self.init_params.depth_mode = depth_mode_enum
        self.init_params.coordinate_units = units_enum
        self.init_params.depth_minimum_distance = float(depth_minimum_distance)
        self.init_params.depth_maximum_distance = float(depth_maximum_distance)
        if self.svo_mode and hasattr(self.init_params, "svo_real_time_mode"):
            # Batch processing must see every recorded frame, so real time
            # replay (which drops frames) is off unless asked for.
            self.init_params.svo_real_time_mode = bool(svo_real_time_mode)

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(
                "Unable to open the ZED camera: {0}. "
                "When reading an SVO, check that the path is correct; when "
                "using the live camera, check that it is plugged in and not "
                "already in use by another process.".format(repr(err))
            )

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        sensing_mode_enum = self._get_sensing_mode_enum()
        if sensing_mode_enum is not None:
            self._set_runtime_param(self.runtime_parameters, "sensing_mode", sensing_mode_enum)

        # Setting the depth confidence parameters when supported
        self._set_runtime_param(self.runtime_parameters, "confidence_threshold", depth_confidence)
        self._set_runtime_param(
            self.runtime_parameters, "textureness_confidence_threshold", depth_confidence
        )

        # Cache camera information for backward and forward compatibility
        self.camera_info = self.zed.get_camera_information()
        self.camera_params = self._get_left_camera_calibration(self.camera_info)
        self.fx = self.camera_params.fx  # Focal length in pixels (x-axis)
        self.fy = self.camera_params.fy  # Focal length in pixels (y-axis)
        self.cx = self.camera_params.cx  # X-coordinate of the principal point
        self.cy = self.camera_params.cy  # Y-coordinate of the principal point

        self.camera_resolution = self._get_camera_resolution(self.camera_info)
        self.camera_fps = self._get_camera_fps(self.camera_info)

        # Enumeration compatibility helpers
        self._confidence_view = self._get_enum_value(sl.VIEW, "CONFIDENCE", "DEPTH")
        self._confidence_measure = self._get_measure("CONFIDENCE", "DEPTH")
        self._mem_cpu = self._get_mem_cpu()

        # declare image, depth, point cloud
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()
        self.img = None
        self.depth_img = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def grab(self):
        """Grab the next frame. Returns the raw SDK error code."""
        return self.zed.grab(self.runtime_parameters)

    def get_image(self):
        """Retrieve the colour image, depth view, point cloud and confidence."""
        # Retrieve left rectified image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_image(self.depth, self._confidence_view)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, self._mem_cpu)
        # Retrieve confidence map.
        self.zed.retrieve_measure(self.confidence_map, self._confidence_measure, self._mem_cpu)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.depth_img = self.depth.get_data()

    def get_number_of_frames(self):
        """Number of frames in the SVO, or ``-1`` for a live camera."""
        for name in ("get_svo_number_of_frames", "get_number_of_frames"):
            getter = getattr(self.zed, name, None)
            if callable(getter):
                return getter()
        return -1

    def get_position(self):
        """Current frame index inside the SVO, or ``-1`` for a live camera."""
        getter = getattr(self.zed, "get_svo_position", None)
        return getter() if callable(getter) else -1

    def set_position(self, index):
        """Seek to `index` inside the SVO (no-op for a live camera)."""
        setter = getattr(self.zed, "set_svo_position", None)
        if callable(setter):
            setter(int(index))

    def get_timestamp_ns(self):
        """Timestamp of the last grabbed image, in nanoseconds."""
        return self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

    def close(self):
        self.zed.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_information(self):
        width = (
            round(self.camera_resolution.width, 2)
            if hasattr(self.camera_resolution, "width")
            else self.camera_resolution[0]
        )
        height = (
            self.camera_resolution.height
            if hasattr(self.camera_resolution, "height")
            else self.camera_resolution[1]
        )
        print("Resolution: {0}, {1}.".format(width, height))
        print("Camera FPS: {0}".format(self.camera_fps))
        depth_mode = getattr(self.init_params, "depth_mode", "N/A")
        print("Depth mode: {0}.".format(depth_mode))
        sensing_mode = self._get_runtime_sensing_mode(self.runtime_parameters)
        print("Sensing mode: {0}.".format(sensing_mode))
        if self.svo_mode:
            print("Frame count: {0}.\n".format(self.get_number_of_frames()))

    def _log(self, message):
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # SDK compatibility helpers
    # ------------------------------------------------------------------
    def _get_left_camera_calibration(self, camera_info):
        calibration = getattr(camera_info, "calibration_parameters", None)
        if calibration and hasattr(calibration, "left_cam"):
            return calibration.left_cam

        camera_configuration = getattr(camera_info, "camera_configuration", None)
        if camera_configuration is not None:
            calibration = getattr(camera_configuration, "calibration_parameters", None)
            if calibration and hasattr(calibration, "left_cam"):
                return calibration.left_cam
            left_cam = getattr(camera_configuration, "left_cam", None)
            if left_cam is not None:
                return left_cam

        raise AttributeError("Unable to find left camera calibration data in camera information.")

    def _get_camera_resolution(self, camera_info):
        resolution = getattr(camera_info, "camera_resolution", None)
        if resolution is not None:
            return resolution

        camera_configuration = getattr(camera_info, "camera_configuration", None)
        if camera_configuration is not None:
            cfg_resolution = getattr(camera_configuration, "resolution", None)
            if cfg_resolution is not None:
                return cfg_resolution

        return (self.init_params.camera_resolution.width, self.init_params.camera_resolution.height)

    def _get_camera_fps(self, camera_info):
        if hasattr(camera_info, "camera_fps"):
            return camera_info.camera_fps

        camera_configuration = getattr(camera_info, "camera_configuration", None)
        if camera_configuration is not None and hasattr(camera_configuration, "fps"):
            return camera_configuration.fps

        return self.init_params.camera_fps

    def _get_enum_value(self, enum_cls, preferred_name, fallback_name):
        if hasattr(enum_cls, preferred_name):
            return getattr(enum_cls, preferred_name)
        if hasattr(enum_cls, fallback_name):
            return getattr(enum_cls, fallback_name)
        return None

    def _get_measure(self, preferred_name, fallback_name):
        return self._get_enum_value(sl.MEASURE, preferred_name, fallback_name)

    def _get_mem_cpu(self):
        if hasattr(sl.MEM, "CPU"):
            return sl.MEM.CPU
        if hasattr(sl.MEM, "MEM_CPU"):
            return sl.MEM.MEM_CPU
        raise AttributeError("Unable to find CPU memory enumeration in sl.MEM")

    def _get_runtime_sensing_mode(self, runtime_parameters):
        if hasattr(runtime_parameters, "sensing_mode"):
            return runtime_parameters.sensing_mode
        get_mode = getattr(runtime_parameters, "get_sensing_mode", None)
        if callable(get_mode):
            return get_mode()
        return "N/A"

    def _set_runtime_param(self, runtime_parameters, attr_name, value):
        if hasattr(runtime_parameters, attr_name):
            setattr(runtime_parameters, attr_name, value)
        else:
            setter_name = "set_{0}".format(attr_name)
            setter = getattr(runtime_parameters, setter_name, None)
            if callable(setter):
                setter(value)

    def _get_sensing_mode_enum(self):
        # SENSING_MODE was removed in SDK 4.0; on newer SDKs this returns None
        # and the caller simply skips setting it.
        if not hasattr(sl, "SENSING_MODE"):
            return None
        return self._get_enum_value(sl.SENSING_MODE, "FILL", "STANDARD")


def get_resolution_wh(camera_info):
    """``(width, height)`` of an opened camera, across SDK generations."""
    for holder in (camera_info, getattr(camera_info, "camera_configuration", None)):
        if holder is None:
            continue
        for attr in ("camera_resolution", "resolution"):
            resolution = getattr(holder, attr, None)
            if resolution is not None and hasattr(resolution, "width"):
                return int(resolution.width), int(resolution.height)
    raise AttributeError("Unable to read the camera resolution from camera information.")


def get_camera_fps(camera_info, default=30):
    """Frame rate of an opened camera, across SDK generations."""
    for holder in (camera_info, getattr(camera_info, "camera_configuration", None)):
        if holder is None:
            continue
        for attr in ("camera_fps", "fps"):
            fps = getattr(holder, attr, None)
            if fps:
                return fps
    return default


def set_fill_sensing_mode(runtime_parameters):
    """Enable FILL sensing on SDK 3.x; a no-op on SDK 4.x and newer."""
    if not hasattr(sl, "SENSING_MODE") or not hasattr(runtime_parameters, "sensing_mode"):
        return False
    mode, _ = resolve_enum(sl.SENSING_MODE, ("FILL", "STANDARD"), "sensing mode")
    runtime_parameters.sensing_mode = mode
    return True


def is_end_of_svo(err):
    """True when `err` signals that the end of the SVO file was reached.

    The enum member was renamed between SDK generations
    (``END_OF_SVOFILE_REACHED`` vs ``END_OF_SVO_FILE``), so both the enum and
    its textual form are checked.
    """
    for name in ("END_OF_SVOFILE_REACHED", "END_OF_SVO_FILE", "END_OF_SVOFILE"):
        member = getattr(sl.ERROR_CODE, name, None)
        if member is not None and err == member:
            return True
    text = str(err).upper()
    return "END_OF_SVO" in text or "END_OF_FILE" in text
