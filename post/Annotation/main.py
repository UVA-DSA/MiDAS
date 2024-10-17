import sys
import json
import os
import cv2
os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "ffmpeg"
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSlider,
    QFileDialog, QMessageBox, QSizePolicy, QFrame
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtCore import QRect, QPoint

from PyQt5.QtWidgets import QToolTip
from PyQt5.QtCore import QEvent

from PyQt5.QtWidgets import QToolTip, QMessageBox
from PyQt5.QtCore import QEvent, pyqtSignal

class ColoredSlider(QSlider):
    annotationDoubleClicked = pyqtSignal(int, int)  # Signal for double-click (track, index)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotations = []
        self.setMouseTracking(True)
        self.installEventFilter(self)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for annotation in self.annotations:
            painter.fillRect(annotation['rect'], annotation['color'])

    def update_annotations(self, annotations):
        self.annotations = annotations
        self.update()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove:
            for annotation in self.annotations:
                if annotation['rect'].contains(event.pos()):
                    QToolTip.showText(self.mapToGlobal(event.pos()), annotation['tooltip'])
                    return True
            QToolTip.hideText()
        elif event.type() == QEvent.MouseButtonDblClick:
            for track, annotation in enumerate(self.annotations):
                if annotation['rect'].contains(event.pos()):
                    index = annotation['index']
                    self.annotationDoubleClicked.emit(track, index)
                    return True
        return super().eventFilter(source, event)

class ColorBar(QFrame):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(20, 20)  # Set the size of the color bar

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.color)

class VideoAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")
        self.resize(800, 600)

        # Define colors for each track
        self.track_colors = [
            QColor(255, 0, 0, 128),  # Red for Track 1
            QColor(0, 255, 0, 128),  # Green for Track 2
            QColor(0, 0, 255, 128),  # Blue for Track 3
            QColor(255, 255, 0, 128),  # Yellow for Track 4 (Gestures)
            QColor(255, 0, 255, 128)  # Magenta for Track 5 (Phases)
        ]

        # Initialize media player and video widget
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

        # Set size policy for video widget to expand
        self.videoWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialize variables for annotations
        self.annotations1 = []
        self.current_segment1 = {
            'start': None, 'end': None,
            'verb': None, 'instrument': None, 'target': None
        }

        self.annotations2 = []
        self.current_segment2 = {
            'start': None, 'end': None,
            'verb': None, 'instrument': None, 'target': None
        }

        self.annotations3 = []
        self.current_segment3 = {
            'start': None, 'end': None,
            'verb': None, 'instrument': None, 'target': None
        }

        self.annotations4 = []
        self.current_segment4 = {
            'start': None, 'end': None,
            'gesture': None
        }

        self.annotations5 = []
        self.current_segment5 = {
            'start': None, 'end': None,
            'phase': None
        }

        # Load labels from config files
        self.load_labels()

        # Set up the user interface
        self.init_ui()

        # Connect the new signal
        self.positionSlider.annotationDoubleClicked.connect(self.remove_annotation)

        self.frame_rate = None
        self.total_frames = None

    def load_labels(self):
        # Load labels from JSON config files
        try:
            with open('verb_labels.json', 'r') as f:
                self.verbs = json.load(f)
        except:
            self.verbs = []
        try:
            with open('instrument_labels.json', 'r') as f:
                self.instruments = json.load(f)
        except:
            self.instruments = []
        try:
            with open('target_labels.json', 'r') as f:
                self.targets = json.load(f)
        except:
            self.targets = []
        # Load gestures and phases for Track 4
        try:
            with open('gesture_labels.json', 'r') as f:
                self.gestures = json.load(f)
        except:
            self.gestures = []
        try:
            with open('phase_labels.json', 'r') as f:
                self.phases = json.load(f)
        except:
            self.phases = []

    def init_ui(self):
        # Create control buttons
        self.openButton = QPushButton('Open')
        self.openButton.setFixedWidth(60)
        self.openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton('Play')
        self.playButton.setFixedWidth(60)
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.play_video)

        self.stepBackwardButton = QPushButton('<<')
        self.stepBackwardButton.setFixedWidth(40)
        self.stepBackwardButton.setEnabled(False)
        self.stepBackwardButton.clicked.connect(self.step_backward)

        self.stepForwardButton = QPushButton('>>')
        self.stepForwardButton.setFixedWidth(40)
        self.stepForwardButton.setEnabled(False)
        self.stepForwardButton.clicked.connect(self.step_forward)

        # Replace the existing QSlider with our custom ColoredSlider
        self.positionSlider = ColoredSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)
        self.positionSlider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.positionSlider.setFixedHeight(30)  # Increase height to accommodate multiple tracks

        # Create drop-down menus and buttons for labels for Track 1
        self.verbComboBox1 = QComboBox()
        self.verbComboBox1.addItems(self.verbs)
        self.verbComboBox1.setFixedWidth(150)
        self.instrumentComboBox1 = QComboBox()
        self.instrumentComboBox1.addItems(self.instruments)
        self.instrumentComboBox1.setFixedWidth(350)
        self.targetComboBox1 = QComboBox()
        self.targetComboBox1.addItems(self.targets)
        self.targetComboBox1.setFixedWidth(150)

        self.startButton1 = QPushButton('<')
        self.startButton1.setFixedWidth(30)
        self.startButton1.setEnabled(False)
        self.startButton1.clicked.connect(self.start_segment1)

        self.endButton1 = QPushButton('>')
        self.endButton1.setFixedWidth(30)
        self.endButton1.setEnabled(False)
        self.endButton1.clicked.connect(self.end_segment1)

        self.recordingIndicator1 = QLabel()
        self.update_recording_indicator(1, recording=False)

        # Create drop-down menus and buttons for labels for Track 2
        self.verbComboBox2 = QComboBox()
        self.verbComboBox2.addItems(self.verbs)
        self.verbComboBox2.setFixedWidth(150)
        self.instrumentComboBox2 = QComboBox()
        self.instrumentComboBox2.addItems(self.instruments)
        self.instrumentComboBox2.setFixedWidth(350)
        self.targetComboBox2 = QComboBox()
        self.targetComboBox2.addItems(self.targets)
        self.targetComboBox2.setFixedWidth(150)

        self.startButton2 = QPushButton('<')
        self.startButton2.setFixedWidth(30)
        self.startButton2.setEnabled(False)
        self.startButton2.clicked.connect(self.start_segment2)

        self.endButton2 = QPushButton('>')
        self.endButton2.setFixedWidth(30)
        self.endButton2.setEnabled(False)
        self.endButton2.clicked.connect(self.end_segment2)

        self.recordingIndicator2 = QLabel()
        self.update_recording_indicator(2, recording=False)

        # Create drop-down menus and buttons for labels for Track 3
        self.verbComboBox3 = QComboBox()
        self.verbComboBox3.addItems(self.verbs)
        self.verbComboBox3.setFixedWidth(150)
        self.instrumentComboBox3 = QComboBox()
        self.instrumentComboBox3.addItems(self.instruments)
        self.instrumentComboBox3.setFixedWidth(350)
        self.targetComboBox3 = QComboBox()
        self.targetComboBox3.addItems(self.targets)
        self.targetComboBox3.setFixedWidth(150)

        self.startButton3 = QPushButton('<')
        self.startButton3.setFixedWidth(30)
        self.startButton3.setEnabled(False)
        self.startButton3.clicked.connect(self.start_segment3)

        self.endButton3 = QPushButton('>')
        self.endButton3.setFixedWidth(30)
        self.endButton3.setEnabled(False)
        self.endButton3.clicked.connect(self.end_segment3)

        self.recordingIndicator3 = QLabel()
        self.update_recording_indicator(3, recording=False)

        # Create drop-down menus and buttons for labels for Track 4 (Gestures)
        self.gestureComboBox4 = QComboBox()
        self.gestureComboBox4.addItems(self.gestures)
        self.gestureComboBox4.setFixedWidth(150)

        self.startButton4 = QPushButton('<')
        self.startButton4.setFixedWidth(30)
        self.startButton4.setEnabled(False)
        self.startButton4.clicked.connect(self.start_segment4)

        self.endButton4 = QPushButton('>')
        self.endButton4.setFixedWidth(30)
        self.endButton4.setEnabled(False)
        self.endButton4.clicked.connect(self.end_segment4)

        self.recordingIndicator4 = QLabel()
        self.update_recording_indicator(4, recording=False)

        # Create drop-down menus and buttons for labels for Track 5 (Phases)
        self.phaseComboBox5 = QComboBox()
        self.phaseComboBox5.addItems(self.phases)
        self.phaseComboBox5.setFixedWidth(150)

        self.startButton5 = QPushButton('<')
        self.startButton5.setFixedWidth(30)
        self.startButton5.setEnabled(False)
        self.startButton5.clicked.connect(self.start_segment5)

        self.endButton5 = QPushButton('>')
        self.endButton5.setFixedWidth(30)
        self.endButton5.setEnabled(False)
        self.endButton5.clicked.connect(self.end_segment5)

        self.recordingIndicator5 = QLabel()
        self.update_recording_indicator(5, recording=False)

        self.saveAnnotationsButton = QPushButton('Save')
        self.saveAnnotationsButton.setFixedWidth(60)
        self.saveAnnotationsButton.clicked.connect(self.save_annotations)

        # Control layout
        controlLayout = QHBoxLayout()
        controlLayout.setSpacing(5)
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stepBackwardButton)
        controlLayout.addWidget(self.stepForwardButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.saveAnnotationsButton)

        # Create a separator line
        separatorLine1 = QFrame()
        separatorLine1.setFrameShape(QFrame.HLine)
        separatorLine1.setFrameShadow(QFrame.Sunken)

        # Label grid layout for compact arrangement
        labelGrid = QGridLayout()
        labelGrid.setSpacing(5)

        # Row 0 - Headers
        labelGrid.addWidget(QLabel('Track'), 0, 0)
        labelGrid.addWidget(QLabel('Verb / Gesture'), 0, 1)
        labelGrid.addWidget(QLabel('Instrument / Phase'), 0, 2)
        labelGrid.addWidget(QLabel('Target'), 0, 3)
        labelGrid.addWidget(QLabel('Start'), 0, 4)
        labelGrid.addWidget(QLabel('End'), 0, 5)
        labelGrid.addWidget(QLabel('Status'), 0, 6)

        # Row 1 - Track 1
        track1Layout = QHBoxLayout()
        track1Layout.addWidget(ColorBar(self.track_colors[0]))
        track1Layout.addWidget(QLabel('1'))
        labelGrid.addLayout(track1Layout, 1, 0)
        labelGrid.addWidget(self.verbComboBox1, 1, 1)
        labelGrid.addWidget(self.instrumentComboBox1, 1, 2)
        labelGrid.addWidget(self.targetComboBox1, 1, 3)
        labelGrid.addWidget(self.startButton1, 1, 4)
        labelGrid.addWidget(self.endButton1, 1, 5)
        labelGrid.addWidget(self.recordingIndicator1, 1, 6)

        # Row 2 - Track 2
        track2Layout = QHBoxLayout()
        track2Layout.addWidget(ColorBar(self.track_colors[1]))
        track2Layout.addWidget(QLabel('2'))
        labelGrid.addLayout(track2Layout, 2, 0)
        labelGrid.addWidget(self.verbComboBox2, 2, 1)
        labelGrid.addWidget(self.instrumentComboBox2, 2, 2)
        labelGrid.addWidget(self.targetComboBox2, 2, 3)
        labelGrid.addWidget(self.startButton2, 2, 4)
        labelGrid.addWidget(self.endButton2, 2, 5)
        labelGrid.addWidget(self.recordingIndicator2, 2, 6)

        # Row 3 - Track 3
        track3Layout = QHBoxLayout()
        track3Layout.addWidget(ColorBar(self.track_colors[2]))
        track3Layout.addWidget(QLabel('3'))
        labelGrid.addLayout(track3Layout, 3, 0)
        labelGrid.addWidget(self.verbComboBox3, 3, 1)
        labelGrid.addWidget(self.instrumentComboBox3, 3, 2)
        labelGrid.addWidget(self.targetComboBox3, 3, 3)
        labelGrid.addWidget(self.startButton3, 3, 4)
        labelGrid.addWidget(self.endButton3, 3, 5)
        labelGrid.addWidget(self.recordingIndicator3, 3, 6)

        # Row 4 - Track 4 (Gestures)
        track4Layout = QHBoxLayout()
        track4Layout.addWidget(ColorBar(self.track_colors[3]))
        track4Layout.addWidget(QLabel('4'))
        labelGrid.addLayout(track4Layout, 4, 0)
        labelGrid.addWidget(self.gestureComboBox4, 4, 1)
        labelGrid.addWidget(QLabel(''), 4, 2)  # Empty cell for 'Instrument'
        labelGrid.addWidget(QLabel(''), 4, 3)  # Empty cell for 'Target'
        labelGrid.addWidget(self.startButton4, 4, 4)
        labelGrid.addWidget(self.endButton4, 4, 5)
        labelGrid.addWidget(self.recordingIndicator4, 4, 6)

        # Row 5 - Track 5 (Phases)
        track5Layout = QHBoxLayout()
        track5Layout.addWidget(ColorBar(self.track_colors[4]))
        track5Layout.addWidget(QLabel('5'))
        labelGrid.addLayout(track5Layout, 5, 0)
        labelGrid.addWidget(self.phaseComboBox5, 5, 1)
        labelGrid.addWidget(QLabel(''), 5, 2)  # Empty cell for 'Instrument'
        labelGrid.addWidget(QLabel(''), 5, 3)  # Empty cell for 'Target'
        labelGrid.addWidget(self.startButton5, 5, 4)
        labelGrid.addWidget(self.endButton5, 5, 5)
        labelGrid.addWidget(self.recordingIndicator5, 5, 6)

        # Main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(separatorLine1)
        mainLayout.addLayout(labelGrid)
        mainLayout.setStretchFactor(self.videoWidget, 1)
        mainLayout.setStretchFactor(controlLayout, 0)
        mainLayout.setStretchFactor(labelGrid, 0)

        self.setLayout(mainLayout)

        # Connect media player signals
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def update_recording_indicator(self, track_number, recording):
        if track_number == 1:
            indicator = self.recordingIndicator1
        elif track_number == 2:
            indicator = self.recordingIndicator2
        elif track_number == 3:
            indicator = self.recordingIndicator3
        elif track_number == 4:
            indicator = self.recordingIndicator4
        elif track_number == 5:
            indicator = self.recordingIndicator5
        else:
            return  # Invalid track number

        if recording:
            indicator.setText('●')
            indicator.setStyleSheet('color: green;')
        else:
            indicator.setText('●')
            indicator.setStyleSheet('color: red;')

    def open_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.stepBackwardButton.setEnabled(True)
            self.stepForwardButton.setEnabled(True)
            self.startButton1.setEnabled(True)
            self.endButton1.setEnabled(True)
            self.startButton2.setEnabled(True)
            self.endButton2.setEnabled(True)
            self.startButton3.setEnabled(True)
            self.endButton3.setEnabled(True)
            self.startButton4.setEnabled(True)
            self.endButton4.setEnabled(True)
            self.startButton5.setEnabled(True)
            self.endButton5.setEnabled(True)

            # Get video information using OpenCV
            video = cv2.VideoCapture(fileName)
            self.frame_rate = video.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()

            if self.frame_rate is None or self.frame_rate == 0:
                QMessageBox.warning(self, 'Warning', 'Could not determine video frame rate. Using default of 30 fps.')
                self.frame_rate = 30.0

            print(f"Loaded video with frame rate: {self.frame_rate} fps and {self.total_frames} total frames")

    def play_video(self):
        # Play or pause the video
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setText('Play')
        else:
            self.mediaPlayer.play()
            self.playButton.setText('Pause')

    def position_changed(self, position):
        # Update the seeker bar position
        self.positionSlider.setValue(position)
        current_frame = self.position_to_frame(position)

    def duration_changed(self, duration):
        # Set the seeker bar range
        self.positionSlider.setRange(0, duration)
        self.update_slider_annotations()

    def set_position(self, position):
        # Set the media player's position
        self.mediaPlayer.setPosition(position)
        # current_frame = self.position_to_frame(position)
        # QToolTip.showText(self.positionSlider.mapToGlobal(QPoint(0, 0)), f"Frame: {current_frame}")

    def position_to_frame(self, position):
        if self.frame_rate is None or self.frame_rate == 0:
            return 0
        return int((position / 1000) * self.frame_rate)

    def frame_to_position(self, frame):
        if self.frame_rate is None or self.frame_rate == 0:
            return 0
        return int((frame / self.frame_rate) * 1000)

    def step_backward(self):
        current_frame = self.position_to_frame(self.mediaPlayer.position())
        new_frame = max(0, current_frame - 5)
        self.mediaPlayer.setPosition(self.frame_to_position(new_frame))

    def step_forward(self):
        current_frame = self.position_to_frame(self.mediaPlayer.position())
        new_frame = min(self.total_frames - 1, current_frame + 5)
        self.mediaPlayer.setPosition(self.frame_to_position(new_frame))

    def start_segment1(self):
        # Mark the start of a segment for Track 1
        self.current_segment1['start'] = self.position_to_frame(self.mediaPlayer.position())
        self.update_recording_indicator(1, recording=True)
        self.update_slider_annotations()

    def end_segment1(self):
        # Mark the end of a segment for Track 1 and save the annotation
        self.current_segment1['end'] = self.position_to_frame(self.mediaPlayer.position())
        self.current_segment1['verb'] = self.verbComboBox1.currentText()
        self.current_segment1['instrument'] = self.instrumentComboBox1.currentText()
        self.current_segment1['target'] = self.targetComboBox1.currentText()

        if self.current_segment1['start'] is not None and self.current_segment1['end'] is not None:
            if self.current_segment1['end'] > self.current_segment1['start']:
                self.annotations1.append(self.current_segment1.copy())
                # Reset current segment
                self.current_segment1 = {
                    'start': None, 'end': None,
                    'verb': None, 'instrument': None, 'target': None
                }
                self.update_recording_indicator(1, recording=False)
                self.update_slider_annotations()
            else:
                QMessageBox.warning(self, 'Warning', 'Track 1: End frame must be after start frame.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 1: Start and end frames must be set.')

    def start_segment2(self):
        # Mark the start of a segment for Track 2
        self.current_segment2['start'] = self.position_to_frame(self.mediaPlayer.position())
        self.update_recording_indicator(2, recording=True)
        self.update_slider_annotations()

    def end_segment2(self):
        # Mark the end of a segment for Track 2 and save the annotation
        self.current_segment2['end'] = self.position_to_frame(self.mediaPlayer.position())
        self.current_segment2['verb'] = self.verbComboBox2.currentText()
        self.current_segment2['instrument'] = self.instrumentComboBox2.currentText()
        self.current_segment2['target'] = self.targetComboBox2.currentText()

        if self.current_segment2['start'] is not None and self.current_segment2['end'] is not None:
            if self.current_segment2['end'] > self.current_segment2['start']:
                self.annotations2.append(self.current_segment2.copy())
                # Reset current segment
                self.current_segment2 = {
                    'start': None, 'end': None,
                    'verb': None, 'instrument': None, 'target': None
                }
                self.update_recording_indicator(2, recording=False)
                self.update_slider_annotations()
            else:
                QMessageBox.warning(self, 'Warning', 'Track 2: End frame must be after start frame.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 2: Start and end frames must be set.')

    def start_segment3(self):
        # Mark the start of a segment for Track 3
        self.current_segment3['start'] = self.position_to_frame(self.mediaPlayer.position())
        self.update_recording_indicator(3, recording=True)
        self.update_slider_annotations()

    def end_segment3(self):
        # Mark the end of a segment for Track 3 and save the annotation
        self.current_segment3['end'] = self.position_to_frame(self.mediaPlayer.position())
        self.current_segment3['verb'] = self.verbComboBox3.currentText()
        self.current_segment3['instrument'] = self.instrumentComboBox3.currentText()
        self.current_segment3['target'] = self.targetComboBox3.currentText()

        if self.current_segment3['start'] is not None and self.current_segment3['end'] is not None:
            if self.current_segment3['end'] > self.current_segment3['start']:
                self.annotations3.append(self.current_segment3.copy())
                # Reset current segment
                self.current_segment3 = {
                    'start': None, 'end': None,
                    'verb': None, 'instrument': None, 'target': None
                }
                self.update_recording_indicator(3, recording=False)
                self.update_slider_annotations()
            else:
                QMessageBox.warning(self, 'Warning', 'Track 3: End frame must be after start frame.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 3: Start and end frames must be set.')

    def start_segment4(self):
        # Mark the start of a segment for Track 4 (Gestures)
        self.current_segment4['start'] = self.position_to_frame(self.mediaPlayer.position())
        self.update_recording_indicator(4, recording=True)
        self.update_slider_annotations()

    def end_segment4(self):
        # Mark the end of a segment for Track 4 (Gestures) and save the annotation
        self.current_segment4['end'] = self.position_to_frame(self.mediaPlayer.position())
        self.current_segment4['gesture'] = self.gestureComboBox4.currentText()

        if self.current_segment4['start'] is not None and self.current_segment4['end'] is not None:
            if self.current_segment4['end'] > self.current_segment4['start']:
                self.annotations4.append(self.current_segment4.copy())
                # Reset current segment
                self.current_segment4 = {
                    'start': None, 'end': None,
                    'gesture': None
                }
                self.update_recording_indicator(4, recording=False)
                self.update_slider_annotations()
            else:
                QMessageBox.warning(self, 'Warning', 'Track 4: End frame must be after start frame.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 4: Start and end frames must be set.')

    def start_segment5(self):
        # Mark the start of a segment for Track 5 (Phases)
        self.current_segment5['start'] = self.position_to_frame(self.mediaPlayer.position())
        self.update_recording_indicator(5, recording=True)
        self.update_slider_annotations()

    def end_segment5(self):
        # Mark the end of a segment for Track 5 (Phases) and save the annotation
        self.current_segment5['end'] = self.position_to_frame(self.mediaPlayer.position())
        self.current_segment5['phase'] = self.phaseComboBox5.currentText()

        if self.current_segment5['start'] is not None and self.current_segment5['end'] is not None:
            if self.current_segment5['end'] > self.current_segment5['start']:
                self.annotations5.append(self.current_segment5.copy())
                # Reset current segment
                self.current_segment5 = {
                    'start': None, 'end': None,
                    'phase': None
                }
                self.update_recording_indicator(5, recording=False)
                self.update_slider_annotations()
            else:
                QMessageBox.warning(self, 'Warning', 'Track 5: End frame must be after start frame.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 5: Start and end frames must be set.')

    def save_annotations(self):
        # Default file names for each track
        default_file_names = [
            "annotations_track1.json",
            "annotations_track2.json",
            "annotations_track3.json",
            "annotations_track4_gestures.json",
            "annotations_track5_phases.json"
        ]

        saved_files = []

        # Save annotations for all tracks
        for i, annotations in enumerate([self.annotations1, self.annotations2, self.annotations3, self.annotations4, self.annotations5], 1):
            fileName = default_file_names[i-1]
            
            # Check if file already exists
            if os.path.exists(fileName):
                reply = QMessageBox.question(self, 'Overwrite File',
                                             f"The file {fileName} already exists. Do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.No:
                    print(f"Skipping Track {i} annotations as per user request.")
                    continue

            # Save the file
            with open(fileName, 'w') as f:
                json.dump(annotations, f, indent=4)
            print(f"Annotations for Track {i} saved to {fileName}")
            saved_files.append(fileName)

        # Show completion dialog
        if saved_files:
            message = "Annotations saved to:\n" + "\n".join(saved_files)
            QMessageBox.information(self, 'Annotations Saved', message)
        else:
            QMessageBox.information(self, 'No Annotations Saved', "No annotation files were saved.")

    def update_slider_annotations(self):
        slider_width = self.positionSlider.width()
        
        if self.total_frames == 0:
            return

        annotations = []
        for track, track_annotations in enumerate([self.annotations1, self.annotations2, self.annotations3, self.annotations4, self.annotations5]):
            for index, annotation in enumerate(track_annotations):
                start_pos = int((annotation['start'] / self.total_frames) * slider_width)
                end_pos = int((annotation['end'] / self.total_frames) * slider_width)
                height = 5
                y_pos = track * height

                rect = QRect(start_pos, y_pos, end_pos - start_pos, height)
                
                if track < 3:  # Tracks 1, 2, 3
                    tooltip = f"Start Frame: {annotation['start']}\nEnd Frame: {annotation['end']}\nVerb: {annotation['verb']}\nInstrument: {annotation['instrument']}\nTarget: {annotation['target']}"
                elif track == 3:  # Track 4 (Gestures)
                    tooltip = f"Start Frame: {annotation['start']}\nEnd Frame: {annotation['end']}\nGesture: {annotation['gesture']}"
                else:  # Track 5 (Phases)
                    tooltip = f"Start Frame: {annotation['start']}\nEnd Frame: {annotation['end']}\nPhase: {annotation['phase']}"

                annotations.append({
                    'rect': rect,
                    'color': self.track_colors[track],
                    'tooltip': tooltip,
                    'index': index,
                })

        self.positionSlider.update_annotations(annotations)

    def remove_annotation(self, track, index):
        track_annotations = [self.annotations1, self.annotations2, self.annotations3, self.annotations4, self.annotations5][track]
        
        if 0 <= index < len(track_annotations):
            reply = QMessageBox.question(self, 'Remove Annotation', 
                                                         f"Are you sure you want to remove this annotation from Track {track + 1}?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                del track_annotations[index]
                self.update_slider_annotations()
                QMessageBox.information(self, 'Annotation Removed',
                                        f"Annotation removed from Track {track + 1}")
        else:
            QMessageBox.warning(self, 'Error', 'Invalid annotation index')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_slider_annotations()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoAnnotationApp()
    window.show()
    sys.exit(app.exec_())