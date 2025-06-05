import sys
import os
import cv2
import numpy as np
import face_recognition
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QHBoxLayout,
    QVBoxLayout, QWidget, QFileDialog, QStackedWidget
)
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Face Recognition (Unsupervised)")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #2B2B2B; color: white;")

        # Create a stacked widget to handle page navigation
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Initialize both pages
        self.main_widget = QWidget()
        self.init_main_page()
        self.settings_widget = QWidget()
        self.init_settings_page()

        self.stack.addWidget(self.main_widget)
        self.stack.addWidget(self.settings_widget)
        self.stack.setCurrentWidget(self.main_widget)

        # Initialize video capture (default webcam)
        self.cap = cv2.VideoCapture(0)
        self.media_file_mode = False  # 'video' or 'image'

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Feature flags
        self.motion_tracking_enabled = False
        self.face_detection_enabled = False  # YOLO-based face detection toggle
        self.face_recognition_enabled = False  # unsupervised face recognition toggle

        # Background subtractor for motion tracking
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

        # Load YOLOv11 face detection model on CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.yolo_model = YOLO("yolo11x-pose.pt").to(self.device)
        except Exception as e:
            print("Error loading YOLOv11 model:", e)
            self.yolo_model = None

        # Global list to accumulate face embeddings for unsupervised recognition
        self.all_face_embeddings = []

    def init_main_page(self):
        # Video display label
        self.video_label = QLabel()
        self.video_label.setFixedSize(1280, 720)

        # Buttons for toggling features
        self.btn_toggle_motion = QPushButton("Motion Tracking (OFF)")
        self.btn_toggle_face_det = QPushButton("Face Detection (OFF)")
        self.btn_toggle_face_rec = QPushButton("Unsupervised Face Recognition (OFF)")
        self.btn_load_media = QPushButton("Load Media")
        self.btn_settings = QPushButton()
        self.btn_exit = QPushButton("Exit")

        # Set icon for settings button
        self.btn_settings.setIcon(QIcon("settings.svg"))
        self.btn_settings.setFixedSize(40, 40)

        # Button styling
        button_style = "background-color: #444; color: white; padding: 10px; font-size: 14px;"
        for btn in [self.btn_toggle_motion, self.btn_toggle_face_det, self.btn_toggle_face_rec, self.btn_load_media]:
            btn.setStyleSheet(button_style)
        self.btn_exit.setStyleSheet("background-color: #ff4444; color: white; padding: 10px; font-size: 14px;")

        # Connect button functions
        self.btn_toggle_motion.clicked.connect(self.toggle_motion)
        self.btn_toggle_face_det.clicked.connect(self.toggle_face_detection)
        self.btn_toggle_face_rec.clicked.connect(self.toggle_face_recognition)
        self.btn_load_media.clicked.connect(self.load_media)
        self.btn_settings.clicked.connect(self.show_settings_page)
        self.btn_exit.clicked.connect(self.close)

        # Layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle_motion)
        button_layout.addWidget(self.btn_toggle_face_det)
        button_layout.addWidget(self.btn_toggle_face_rec)
        button_layout.addWidget(self.btn_load_media)
        button_layout.addWidget(self.btn_settings)
        button_layout.addWidget(self.btn_exit)

        # Main layout for the main page
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)
        self.main_widget.setLayout(main_layout)

    def init_settings_page(self):
        # Settings page layout
        layout = QVBoxLayout()
        label = QLabel("Settings Page - Placeholder")
        layout.addWidget(label)

        # Back button to return to the main page
        btn_back = QPushButton("Back")
        btn_back.setStyleSheet("background-color: #444; color: white; padding: 10px; font-size: 14px;")
        btn_back.clicked.connect(self.show_main_page)
        layout.addWidget(btn_back)

        self.settings_widget.setLayout(layout)

    def show_settings_page(self):
        self.stack.setCurrentWidget(self.settings_widget)

    def show_main_page(self):
        self.stack.setCurrentWidget(self.main_widget)

    def toggle_motion(self):
        self.motion_tracking_enabled = not self.motion_tracking_enabled
        status = "ON" if self.motion_tracking_enabled else "OFF"
        self.btn_toggle_motion.setText(f"Motion Tracking ({status})")

    def toggle_face_detection(self):
        self.face_detection_enabled = not self.face_detection_enabled
        status = "ON" if self.face_detection_enabled else "OFF"
        self.btn_toggle_face_det.setText(f"Face Detection ({status})")

    def toggle_face_recognition(self):
        self.face_recognition_enabled = not self.face_recognition_enabled
        status = "ON" if self.face_recognition_enabled else "OFF"
        self.btn_toggle_face_rec.setText(f"Unsupervised Face Rec ({status})")

    def load_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Media File", "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All Files (*)"
        )
        if file_path:
            if self.cap.isOpened():
                self.cap.release()
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.frame_image = cv2.imread(file_path)
                self.media_file_mode = 'image'
            else:
                self.cap = cv2.VideoCapture(file_path)
                self.media_file_mode = 'video'

    def update_frame(self):
        # Get current frame
        if self.media_file_mode == 'image':
            frame = self.frame_image.copy()
        else:
            ret, frame = self.cap.read()
            if not ret:
                if self.media_file_mode == 'video':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                else:
                    return

        frame = cv2.resize(frame, (1280, 720))
        display_frame = frame.copy()

        # Motion tracking (if enabled)
        if self.motion_tracking_enabled:
            mask = self.bg_subtractor.apply(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Face detection using YOLOv11 (if enabled)
        detected_face_boxes = []
        if self.face_detection_enabled and self.yolo_model is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(rgb_frame)
            for box in results[0].boxes:
                coords = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, coords)
                detected_face_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,0,255), 2)

        # Unsupervised face recognition (clustering)
        if self.face_recognition_enabled and self.face_detection_enabled:
            new_face_indices = []
            for (x1, y1, x2, y2) in detected_face_boxes:
                face_roi = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(face_rgb)
                if encodings:
                    encoding = encodings[0]
                    idx = len(self.all_face_embeddings)
                    self.all_face_embeddings.append(encoding)
                    new_face_indices.append(idx)
            if self.all_face_embeddings:
                embeddings_array = np.array(self.all_face_embeddings)
                clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(embeddings_array)
                labels = clustering.labels_
                for idx in new_face_indices:
                    label = labels[idx]
                    box = detected_face_boxes[new_face_indices.index(idx)]
                    x1, y1, x2, y2 = box
                    cv2.putText(display_frame, f"Cluster {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
