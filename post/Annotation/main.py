import sys
import json
import os
os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "ffmpeg"
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider,
    QFileDialog, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class VideoAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")
        self.resize(800, 600)

        # Initialize media player and video widget
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

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

        # Load labels from config files
        self.load_labels()

        # Set up the user interface
        self.init_ui()

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

    def init_ui(self):
        # Create control buttons
        self.openButton = QPushButton('Open Video')
        self.openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton('Play')
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.play_video)

        self.stepBackwardButton = QPushButton('Step Backward 5 Frames')
        self.stepBackwardButton.setEnabled(False)
        self.stepBackwardButton.clicked.connect(self.step_backward)

        self.stepForwardButton = QPushButton('Step Forward 5 Frames')
        self.stepForwardButton.setEnabled(False)
        self.stepForwardButton.clicked.connect(self.step_forward)

        # Track 1 segment buttons
        self.startSegmentButton1 = QPushButton('Start Segment Track 1')
        self.startSegmentButton1.setEnabled(False)
        self.startSegmentButton1.clicked.connect(self.start_segment1)

        self.endSegmentButton1 = QPushButton('End Segment Track 1')
        self.endSegmentButton1.setEnabled(False)
        self.endSegmentButton1.clicked.connect(self.end_segment1)

        # Track 2 segment buttons
        self.startSegmentButton2 = QPushButton('Start Segment Track 2')
        self.startSegmentButton2.setEnabled(False)
        self.startSegmentButton2.clicked.connect(self.start_segment2)

        self.endSegmentButton2 = QPushButton('End Segment Track 2')
        self.endSegmentButton2.setEnabled(False)
        self.endSegmentButton2.clicked.connect(self.end_segment2)

        self.saveAnnotationsButton = QPushButton('Save Annotations')
        self.saveAnnotationsButton.clicked.connect(self.save_annotations)

        # Create seeker bar
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)

        # Create drop-down menus for labels for Track 1
        self.verbComboBox1 = QComboBox()
        self.verbComboBox1.addItems(self.verbs)
        self.instrumentComboBox1 = QComboBox()
        self.instrumentComboBox1.addItems(self.instruments)
        self.targetComboBox1 = QComboBox()
        self.targetComboBox1.addItems(self.targets)

        # Create drop-down menus for labels for Track 2
        self.verbComboBox2 = QComboBox()
        self.verbComboBox2.addItems(self.verbs)
        self.instrumentComboBox2 = QComboBox()
        self.instrumentComboBox2.addItems(self.instruments)
        self.targetComboBox2 = QComboBox()
        self.targetComboBox2.addItems(self.targets)

        # Layout configurations
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stepBackwardButton)
        controlLayout.addWidget(self.stepForwardButton)
        controlLayout.addWidget(self.positionSlider)

        # Segment buttons layout
        segmentLayout = QHBoxLayout()
        # Track 1 buttons
        segmentLayout.addWidget(self.startSegmentButton1)
        segmentLayout.addWidget(self.endSegmentButton1)
        # Track 2 buttons
        segmentLayout.addWidget(self.startSegmentButton2)
        segmentLayout.addWidget(self.endSegmentButton2)

        # Label layouts for Track 1
        labelLayout1 = QHBoxLayout()
        labelLayout1.addWidget(QLabel('Track 1 - Verb:'))
        labelLayout1.addWidget(self.verbComboBox1)
        labelLayout1.addWidget(QLabel('Instrument:'))
        labelLayout1.addWidget(self.instrumentComboBox1)
        labelLayout1.addWidget(QLabel('Target:'))
        labelLayout1.addWidget(self.targetComboBox1)

        # Label layouts for Track 2
        labelLayout2 = QHBoxLayout()
        labelLayout2.addWidget(QLabel('Track 2 - Verb:'))
        labelLayout2.addWidget(self.verbComboBox2)
        labelLayout2.addWidget(QLabel('Instrument:'))
        labelLayout2.addWidget(self.instrumentComboBox2)
        labelLayout2.addWidget(QLabel('Target:'))
        labelLayout2.addWidget(self.targetComboBox2)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)
        mainLayout.addLayout(controlLayout)
        mainLayout.addLayout(segmentLayout)
        mainLayout.addLayout(labelLayout1)
        mainLayout.addLayout(labelLayout2)
        mainLayout.addWidget(self.saveAnnotationsButton)

        self.setLayout(mainLayout)

        # Connect media player signals
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        # Open a file dialog to select a video file
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.stepBackwardButton.setEnabled(True)
            self.stepForwardButton.setEnabled(True)
            self.startSegmentButton1.setEnabled(True)
            self.endSegmentButton1.setEnabled(True)
            self.startSegmentButton2.setEnabled(True)
            self.endSegmentButton2.setEnabled(True)

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

    def duration_changed(self, duration):
        # Set the seeker bar range
        self.positionSlider.setRange(0, duration)

    def set_position(self, position):
        # Set the media player's position
        self.mediaPlayer.setPosition(position)

    def get_frame_duration(self):
        # Approximate frame duration (assuming 30 fps)
        fps = 30
        return int(1000 / fps)  # Duration in milliseconds

    def step_backward(self):
        # Step backward by 5 frames
        frame_duration = self.get_frame_duration()
        new_position = self.mediaPlayer.position() - frame_duration * 5
        self.mediaPlayer.setPosition(max(0, int(new_position)))

    def step_forward(self):
        # Step forward by 5 frames
        frame_duration = self.get_frame_duration()
        new_position = self.mediaPlayer.position() + frame_duration * 5
        self.mediaPlayer.setPosition(min(self.mediaPlayer.duration(), int(new_position)))

    def start_segment1(self):
        # Mark the start of a segment for Track 1
        self.current_segment1['start'] = self.mediaPlayer.position()
        QMessageBox.information(self, 'Info', 'Track 1: Segment start position set.')

    def end_segment1(self):
        # Mark the end of a segment for Track 1 and save the annotation
        self.current_segment1['end'] = self.mediaPlayer.position()
        self.current_segment1['verb'] = self.verbComboBox1.currentText()
        self.current_segment1['instrument'] = self.instrumentComboBox1.currentText()
        self.current_segment1['target'] = self.targetComboBox1.currentText()

        if self.current_segment1['start'] is not None and self.current_segment1['end'] is not None:
            if self.current_segment1['end'] > self.current_segment1['start']:
                self.annotations1.append(self.current_segment1.copy())
                QMessageBox.information(self, 'Info', 'Track 1: Segment saved.')
                # Reset current segment
                self.current_segment1 = {
                    'start': None, 'end': None,
                    'verb': None, 'instrument': None, 'target': None
                }
            else:
                QMessageBox.warning(self, 'Warning', 'Track 1: End position must be after start position.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 1: Start and end positions must be set.')

    def start_segment2(self):
        # Mark the start of a segment for Track 2
        self.current_segment2['start'] = self.mediaPlayer.position()
        QMessageBox.information(self, 'Info', 'Track 2: Segment start position set.')

    def end_segment2(self):
        # Mark the end of a segment for Track 2 and save the annotation
        self.current_segment2['end'] = self.mediaPlayer.position()
        self.current_segment2['verb'] = self.verbComboBox2.currentText()
        self.current_segment2['instrument'] = self.instrumentComboBox2.currentText()
        self.current_segment2['target'] = self.targetComboBox2.currentText()

        if self.current_segment2['start'] is not None and self.current_segment2['end'] is not None:
            if self.current_segment2['end'] > self.current_segment2['start']:
                self.annotations2.append(self.current_segment2.copy())
                QMessageBox.information(self, 'Info', 'Track 2: Segment saved.')
                # Reset current segment
                self.current_segment2 = {
                    'start': None, 'end': None,
                    'verb': None, 'instrument': None, 'target': None
                }
            else:
                QMessageBox.warning(self, 'Warning', 'Track 2: End position must be after start position.')
        else:
            QMessageBox.warning(self, 'Warning', 'Track 2: Start and end positions must be set.')

    def save_annotations(self):
        # Save annotations for Track 1
        fileName1, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations for Track 1", "",
            "JSON Files (*.json)"
        )
        if fileName1 != '':
            with open(fileName1, 'w') as f:
                json.dump(self.annotations1, f, indent=4)
            QMessageBox.information(self, 'Info', 'Annotations for Track 1 saved.')

        # Save annotations for Track 2
        fileName2, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations for Track 2", "",
            "JSON Files (*.json)"
        )
        if fileName2 != '':
            with open(fileName2, 'w') as f:
                json.dump(self.annotations2, f, indent=4)
            QMessageBox.information(self, 'Info', 'Annotations for Track 2 saved.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoAnnotationApp()
    window.show()
    sys.exit(app.exec_())
