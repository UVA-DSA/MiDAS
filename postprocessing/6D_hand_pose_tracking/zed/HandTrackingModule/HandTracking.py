"""MediaPipe hand landmark detection lifted into the ZED's metric 3D space.

MediaPipe returns 21 landmarks per hand in normalised image coordinates. The
wrist landmark is looked up in the ZED point cloud to obtain a metric anchor,
and the remaining landmarks are back-projected around that anchor using the
camera intrinsics. From the resulting 21x3 array the palm centroid and a
yaw/pitch/roll orientation are derived, giving a 6D palm pose.
"""

import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyzed.sl as sl

# MediaPipe hand landmark indices used throughout this package.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17
NUM_LANDMARKS = 21

# Skeleton edges used by the matplotlib preview.
HAND_EDGES = [
    (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (5, 9), (1, 0), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class HandTracking(object):
    """Detect hands in a colour image and locate their landmarks in 3D.

    Args:
        maxHands: Maximum number of hands MediaPipe tracks per frame.
        detectionCon: Minimum detection confidence, 0-1. Higher is stricter
            and faster.
        trackCon: Minimum tracking confidence, 0-1.
        modelComplexity: MediaPipe hand model complexity, ``0`` (fast) or
            ``1`` (accurate, the MediaPipe default).
        swapHandedness: MediaPipe labels hands assuming a mirrored, selfie-view
            image. This package feeds it the unmirrored ZED left image, so the
            default mapping is already inverted to compensate. Set this to
            ``True`` if your recording *is* mirrored and the reported left and
            right hands come out the wrong way round.
    """

    def __init__(
        self,
        maxHands=2,
        detectionCon=0.2,
        trackCon=0.9,
        modelComplexity=1,
        swapHandedness=False,
    ):
        self.mp_hands = mp.solutions.hands
        hand_kwargs = dict(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        try:
            self.hands = self.mp_hands.Hands(model_complexity=modelComplexity, **hand_kwargs)
        except TypeError:
            # Older MediaPipe releases do not expose model_complexity.
            self.hands = self.mp_hands.Hands(**hand_kwargs)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.swap_handedness = swapHandedness
        self.time1 = time.time()
        self.wrist = []
        self.results = None

    def findHands(self, img, draw=True):
        """Run hand detection on `img`.

        Set ``draw=False`` to skip the landmark overlay, which is noticeably
        faster when the output is written straight to a CSV.
        """
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Note: the image is intentionally *not* flipped. See swapHandedness.
        self.results = self.hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

        return img

    def find_positions(self, img, pcl, camera_params, draw=True, verbose=True):
        """Return the left and right hand landmarks in metric 3D coordinates.

        Args:
            img: Image the annotations are drawn on (the scripts pass the
                depth view) and whose shape defines the pixel grid.
            pcl: ZED ``XYZRGBA`` point cloud measure, aligned on the left image.
            camera_params: Left camera calibration (``fx``, ``fy``, ``cx``, ``cy``).
            draw: Draw the wrist marker and the Left/Right label on `img`.
            verbose: Print a one-line detection status to stdout.

        Returns:
            ``(left_data, right_data)``, each an ``(N, 3)`` array that is
            ``(21, 3)`` when the hand was fully resolved and empty otherwise.
            Coordinates use the ZED coordinate frame and the unit chosen when
            the camera was opened (metres by default).
        """
        fx = camera_params.fx  # Focal length in pixels (x-axis)
        fy = camera_params.fy  # Focal length in pixels (y-axis)
        cx = camera_params.cx  # X-coordinate of the principal point
        cy = camera_params.cy  # Y-coordinate of the principal point
        h, w, _ = img.shape
        left_data = []
        right_data = []
        if self.results is not None and self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                index = self.results.multi_hand_landmarks.index(landmarks)
                handedness = self.results.multi_handedness[index].classification[0].index
                is_left = (handedness == 1) if not self.swap_handedness else (handedness == 0)
                wrist_position = None
                wrist_landmark_coordinate = None
                X, Y = 0, 0

                for id, landmark in enumerate(landmarks.landmark):

                    # Find the pixel coordinates of the wrist
                    if id == WRIST:
                        wrist_landmark_coordinate = [landmark.x, landmark.y, landmark.z]
                        X, Y = int(landmark.x * w), int(landmark.y * h)

                        if draw:
                            cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)
                        # Use ZED point cloud to estimate 3D position of wrist
                        try:
                            err, point_cloud_value = pcl.get_value(X, Y)
                            if err == sl.ERROR_CODE.SUCCESS and np.all(
                                np.isfinite(point_cloud_value[:3])
                            ):
                                wrist_position = [
                                    point_cloud_value[0],
                                    point_cloud_value[1],
                                    point_cloud_value[2],
                                ]
                            else:
                                # No valid depth on the wrist (occluded, out of
                                # range, or filtered out by the confidence
                                # threshold) so the whole hand is dropped.
                                wrist_position = None
                        except Exception:
                            wrist_position = None
                            continue

                        if wrist_position is None:
                            continue

                    if wrist_position is None or wrist_landmark_coordinate is None:
                        continue

                    x_3d = (
                        wrist_position[0]
                        + (landmark.x * w - wrist_landmark_coordinate[0] * w - cx)
                        * wrist_position[2]
                        / fx
                    )
                    y_3d = (
                        wrist_position[1]
                        + (landmark.y * h - wrist_landmark_coordinate[1] * w - cy)
                        * wrist_position[2]
                        / fy
                    )
                    z_3d = (
                        wrist_position[2]
                        + (landmark.z - wrist_landmark_coordinate[2]) * wrist_position[2]
                    )
                    hand_landmarks_3d = [x_3d, y_3d, z_3d]
                    # append the 3D position of each 3D landmark
                    if is_left:
                        left_data.append(hand_landmarks_3d)
                        if draw:
                            cv2.putText(
                                img, "Left", (X, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            )
                    else:
                        right_data.append(hand_landmarks_3d)
                        if draw:
                            cv2.putText(
                                img, "Right", (X, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            )

        # Convert the data to a numpy array
        left_data = np.array(left_data)
        right_data = np.array(right_data)
        if verbose:
            self.stdout_hand_detection(left_data, right_data)

        return left_data, right_data

    # Kept for backwards compatibility with code written against the original
    # (misspelled) method name.
    findpostion = find_positions

    def calculate_orientation(self, hand_landmarks_3d):
        """Yaw, pitch and roll of the palm plane, in degrees.

        The palm plane is defined by the wrist, the index MCP and the pinky
        MCP. Returns zeros when the hand was not fully resolved.
        """
        if hand_landmarks_3d.shape != (NUM_LANDMARKS, 3):
            return np.zeros((3,))

        # Get the 3D positions of landmarks 0, 5, and 17
        wrist = hand_landmarks_3d[WRIST]
        index = hand_landmarks_3d[INDEX_MCP]
        pinky = hand_landmarks_3d[PINKY_MCP]

        # Compute the vectors between the landmarks
        v1 = np.subtract(index, wrist)
        v2 = np.subtract(pinky, wrist)

        # Compute the normal vector to the plane defined by the landmarks
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0 or not np.isfinite(norm):
            return np.zeros((3,))
        normal = normal / norm

        # Compute the yaw, pitch, and roll angles based on the orientation of the normal vector
        yaw = np.arctan2(normal[1], normal[0])
        pitch = np.arctan2(-normal[2], np.sqrt(normal[0] ** 2 + normal[1] ** 2))
        roll = np.arctan2(
            np.sin(yaw) * v2[0] - np.cos(yaw) * v2[1],
            np.cos(yaw) * v1[1] - np.sin(yaw) * v1[0],
        )

        self.orientation = np.array([yaw, pitch, roll])

        # Convert angles to degrees and return
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def calculate_centroid(self, hand_landmarks_3d):
        """Palm centre: the mean of the wrist, index MCP and pinky MCP."""
        if hand_landmarks_3d.shape != (NUM_LANDMARKS, 3):
            return np.zeros((3,))

        wrist = hand_landmarks_3d[WRIST]
        index = hand_landmarks_3d[INDEX_MCP]
        pinky = hand_landmarks_3d[PINKY_MCP]

        # Compute a middle point of three landmarks
        return (wrist + index + pinky) / 3

    def findNormalizedPosition(self, img, draw=True, verbose=True):
        """Landmarks in MediaPipe's normalised image coordinates (no depth)."""
        left_data = []
        right_data = []
        w, h, _ = img.shape

        if self.results is not None and self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                index = self.results.multi_hand_landmarks.index(landmarks)
                handedness = self.results.multi_handedness[index].classification[0].index
                is_left = (handedness == 1) if not self.swap_handedness else (handedness == 0)
                X, Y = 0, 0
                for id, landmark in enumerate(landmarks.landmark):
                    # Find the pixel coordinates of the wrist
                    if id == WRIST:
                        X, Y = int(landmark.x * h), int(landmark.y * w)
                        if draw:
                            cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)

                    hand_landmarks_3d = [landmark.x, landmark.y, landmark.z]
                    # append the 3D position of each 3D landmark
                    if is_left:
                        left_data.append(hand_landmarks_3d)
                        if draw:
                            cv2.putText(
                                img, "Left", (X, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            )
                    else:
                        right_data.append(hand_landmarks_3d)
                        if draw:
                            cv2.putText(
                                img, "Right", (X, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            )

        left_data = np.array(left_data)
        right_data = np.array(right_data)
        if verbose:
            self.stdout_hand_detection(left_data, right_data)

        return left_data, right_data

    def displayFPS(self, img):
        # Set the time for this frame to the current time.
        self.time2 = time.time()
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (self.time2 - self.time1) > 0:

            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (self.time2 - self.time1)

            # Write the calculated number of frames per second on the frame.
            cv2.putText(
                img,
                'FPS: {}'.format(int(frames_per_second)),
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                3,
            )
            self.time1 = self.time2

        return img

    def stdout_hand_detection(self, left_data, right_data):
        full = (NUM_LANDMARKS, 3)
        left_ok = left_data.shape == full
        right_ok = right_data.shape == full

        if left_ok and right_ok:
            message = "Left and Right hands all 21 landmarks detected"
        elif left_ok:
            message = "Left hand all 21 landmarks detected"
        elif right_ok:
            message = "Right hand all 21 landmarks detected"
        else:
            message = "No hand landmarks detected"

        sys.stdout.write("\r{0:<50}".format(message))
        sys.stdout.flush()

    def plot(self, ax, plt, data, xlim=(-0.5, 0.1), ylim=(-0.5, 0.1), zlim=(0.2, 1.0)):
        """Draw one or two hand skeletons on a matplotlib 3D axis."""
        if data.shape >= (NUM_LANDMARKS, 3):

            # Clear the plot and add new data
            ax.clear()

            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.scatter3D(*zip(*data))

            # Second hand, if present, is offset by 21 landmarks.
            edges2 = [(a + NUM_LANDMARKS, b + NUM_LANDMARKS) for a, b in HAND_EDGES]

            for edge in HAND_EDGES:
                ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')

            if data.shape == (2 * NUM_LANDMARKS, 3):
                for edge in edges2:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='blue')

            # Draw the plot
            plt.draw()
            plt.pause(0.0001)

    def close(self):
        """Release the MediaPipe graph."""
        if getattr(self, "hands", None) is not None:
            self.hands.close()
            self.hands = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
