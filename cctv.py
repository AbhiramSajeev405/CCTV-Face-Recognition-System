import warnings
warnings.simplefilter("ignore", FutureWarning)
import sys
import os
import json
import logging
import cv2
import numpy as np
import sqlite3
import insightface
import pickle
from datetime import datetime
import time
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO


from PyQt5.QtCore import (QThread, pyqtSignal, QTimer, pyqtSlot, Qt)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QGroupBox, QComboBox,
    QStackedWidget, QFrame, QColorDialog, QAction, QFormLayout, QCheckBox, 
    QMessageBox, QInputDialog, QScrollArea
)

#---------------- Logging Configuration ----------------#
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("app.log", mode="a")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

CONFIG_FILE = "application_config.json"

def load_app_config() -> dict:
    """Load both theme and smtp settings."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
            logging.info(f"Application config loaded: {cfg}")
            return cfg
    logging.info("No application config found; using defaults")
    return {}

def save_app_config(config: dict):
    """Persist entire config (theme + smtp)."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Application config saved: {config}")

# ---------------------------------------------- #
# Helper Function: Non-Maximum Suppression (NMS) #
# ---------------------------------------------- #
def non_max_suppression_indices(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]
        suppress = []
        for pos, i in enumerate(idxs):
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / areas[i]
            if overlap > overlapThresh:
                suppress.append(i)
        idxs = np.array([i for i in idxs if i not in suppress])
    return pick

# -----------------------
# Worker Classes
# -----------------------

class YOLOPoseWorker(QThread):
    peopleReady = pyqtSignal(np.ndarray, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self.running = True
        self.model = None  
    
    @pyqtSlot(str)
    def setModelPath(self, model_path):
        try:
            self.model = YOLO(model_path)
            logging.info(f"YOLO Pose model loaded from {model_path}")
        except Exception as e:
            logging.error("Failed to load YOLO Pose model: " + str(e))
            self.model = None
    
    @pyqtSlot(np.ndarray)
    def processFrame(self, frame):
        self.frame = frame
    
    def run(self):
        while self.running:
            if self.frame is not None:
                if self.model is None:
                    self.frame = None 
                    self.msleep(10)
                    continue

                local_frame_copy = self.frame.copy() 
                self.frame = None
                try:
                    results = self.model(local_frame_copy, verbose=False) 

                    annotated_frame = local_frame_copy 
                    people_info_list = []

                    if results and results[0].boxes is not None:
                        annotated_frame = results[0].plot(labels=False) 
                        boxes = results[0].boxes
                        for box in boxes:
                            coords = box.xyxy.cpu().numpy().astype(int)[0]
                            label = "Person" 
                            people_info_list.append((tuple(coords), label))

                    self.peopleReady.emit(annotated_frame, people_info_list)

                except Exception as e:
                    logging.error(f"Error during YOLO processing or plotting: {e}")

            self.msleep(10) # Small sleep regardless of frame processing
    
    def stop(self):
        self.running = False
        self.wait()

class FaceRecognitionWorker(QThread):
    recognitionReady = pyqtSignal(np.ndarray, list)
    
    def __init__(self, batch_size=4, cluster_interval=30, timeout=1.0, parent=None):
        super().__init__(parent)
        self.running = True
        self.batch_size = batch_size
        self.cluster_interval = cluster_interval
        self.timeout = timeout 
        self.last_process_time = time.time()
        self.frame_counter = 0
        self.cached_labels = []
        self.batch_faces = []  
        self.batch_boxes = []  
        self.current_frame = None  
        
        self.db_conn = sqlite3.connect("cctvfaces.db", check_same_thread=False)
        self.create_faces_table()
        
        self.face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    def create_faces_table(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB,
                label INTEGER
            )
        ''')
        self.db_conn.commit()
    
    def insert_face_to_db(self, embedding):
        cursor = self.db_conn.cursor()
        emb_blob = embedding.astype(np.float32).tobytes()
        cursor.execute("INSERT INTO faces (embedding, label) VALUES (?, ?)", (emb_blob, -1))
        self.db_conn.commit()
        return cursor.lastrowid
    
    def update_db_labels(self, all_ids, labels):
        cursor = self.db_conn.cursor()
        for face_id, label in zip(all_ids, labels):
            cursor.execute("UPDATE faces SET label = ? WHERE id = ?", (int(label), face_id))
        self.db_conn.commit()
    
    @pyqtSlot(np.ndarray, list)
    def processFaces(self, frame, boxes):
        self.current_frame = frame.copy()
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = box
                face_roi = frame[y1:y2, x1:x2]
                self.batch_faces.append(face_roi)
                self.batch_boxes.append(box)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_frame)
            for face in faces:
                bbox = tuple(map(int, face.bbox))
                self.batch_boxes.append(bbox)
                face_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                self.batch_faces.append(face_roi)

        if self.batch_boxes:
            keep_idx = non_max_suppression_indices(self.batch_boxes, overlapThresh=0.3)
            self.batch_boxes = [self.batch_boxes[i] for i in keep_idx]
            self.batch_faces = [self.batch_faces[i] for i in keep_idx]
        self.frame_counter += 1
        current_time = time.time()

        if len(self.batch_faces) >= self.batch_size or (self.batch_faces and (current_time - self.last_process_time) > self.timeout):
            self.processBatch()
            self.batch_faces = []
            self.batch_boxes = []
            self.last_process_time = current_time
    
    def processBatch(self):
        results = []
        for roi in self.batch_faces:
            if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                results.append(None)
            else:
                res = self.face_app.get(roi)
                if res:
                    results.append(res[0])
                else:
                    results.append(None)
        new_face_ids = []
        
        for res in results:
            if res is not None:
                embedding = res.embedding
            else:
                embedding = np.zeros(512, dtype=np.float32)
            face_id = self.insert_face_to_db(embedding)
            new_face_ids.append(face_id)
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, embedding FROM faces ORDER BY id ASC")
        rows = cursor.fetchall()
        all_embeddings = []
        all_ids = []
        
        for row in rows:
            fid = row[0]
            emb = np.frombuffer(row[1], dtype=np.float32)
            all_embeddings.append(emb)
            all_ids.append(fid)
        
        if all_embeddings and len(all_embeddings) > 1 and (self.frame_counter % self.cluster_interval == 0):
            embeddings_array = np.array(all_embeddings)
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.6, min_samples=2, metric='cosine').fit(embeddings_array)
            labels = clustering.labels_.tolist()
            self.update_db_labels(all_ids, labels)
            self.cached_labels = labels
        else:
            labels = self.cached_labels if self.cached_labels else [-1] * len(new_face_ids)
        new_labels = []
        
        for new_id in new_face_ids:
            if new_id in all_ids:
                idx = all_ids.index(new_id)
                new_labels.append(self.cached_labels[idx] if idx < len(self.cached_labels) else -1)
            else:
                new_labels.append(-1)
        
        processed_frame = self.current_frame.copy() if self.current_frame is not None else np.zeros((720,1280,3), dtype=np.uint8)
        
        for (box, label) in zip(self.batch_boxes, new_labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(processed_frame, f"ID {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        self.recognitionReady.emit(processed_frame, list(zip(self.batch_boxes, new_labels)))
    
    def run(self):
        while self.running:
            self.msleep(10)
    
    def stop(self):
        self.running = False
        self.db_conn.close()
        self.wait()

class SupervisedFaceRecognitionWorker(QThread):
    recognitionReady = pyqtSignal(np.ndarray, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.frame = None
        self.classifier = None
        self.classifier_path = None
        self.face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    @pyqtSlot(np.ndarray)
    def processFrame(self, frame):
        self.frame = frame
    
    @pyqtSlot(str)
    def loadClassifier(self, classifier_path):
        try:
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)
            self.classifier_path = classifier_path
            logging.info(f"Loaded classifier from {classifier_path}")
        except Exception as e:
            logging.error("Failed to load classifier: " + str(e))
            self.classifier = None
    
    def run(self):
        while self.running:
            if self.frame is not None:
                if self.classifier is None:
                    logging.warning("No classifier loaded; skipping supervised recognition.")
                    self.msleep(10)
                    continue
                current_frame = self.frame.copy()
                rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                faces = self.face_app.get(rgb_frame)
                face_info = []
                for face in faces:
                    bbox = tuple(map(int, face.bbox))
                    embedding = face.embedding
                    try:
                        if isinstance(self.classifier, dict) and "model" in self.classifier:
                            model_to_use = self.classifier["model"]
                        else:
                            model_to_use = self.classifier
                        pred = model_to_use.predict(np.expand_dims(embedding, axis=0))
                        label = str(pred[0])
                    except Exception as e:
                        logging.error("Error during classifier prediction: " + str(e))
                        label = "Unknown"
                    face_info.append((bbox, label))
                    cv2.rectangle(current_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(current_frame, label, (bbox[0], bbox[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self.recognitionReady.emit(current_frame, face_info)
                self.frame = None
            self.msleep(10)
    
    def stop(self):
        self.running = False
        self.wait()

# -----------------------
# GUI Components
# -----------------------


class ModelPathRow(QWidget):
    modelPathLoaded = pyqtSignal(str)
    
    def __init__(self, label_text):
        super().__init__()
        self.initUI(label_text)
    
    def initUI(self, label_text):
        layout = QHBoxLayout()
        self.label = QLabel(label_text)
        self.line_edit = QLineEdit()
        self.btn_browse = QPushButton("Browse...")
        self.btn_load = QPushButton("Load")
        self.status_label = QLabel("")
        
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_load.clicked.connect(self.load_file)
        self.line_edit.textChanged.connect(self.validate_path)
        
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn_browse)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "All Files (*)")
        if file_path:
            self.line_edit.setText(file_path)
            logging.info(f"Browsing file: {file_path}")
    
    def validate_path(self):
        path = self.line_edit.text().strip()
        if path and not os.path.exists(path):
            self.status_label.setText("Invalid path")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            logging.info("Invalid file path entered")
        else:
            self.status_label.setText("")
    
    def load_file(self):
        path = self.line_edit.text().strip()
        if os.path.exists(path):
            filename = os.path.basename(path)
            self.status_label.setText(f"Loaded: {filename}")
            self.status_label.setStyleSheet("color: green; font-size: 10px;")
            logging.info(f"Model loaded: {filename}")
            self.modelPathLoaded.emit(path)
        else:
            self.status_label.setText("File not found")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            logging.info("Model load failed: file not found")

class InputSourcePanel(QWidget):
    mediaSelected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Camera", "Media", "Network Stream"])
        self.combo_source.currentIndexChanged.connect(self.change_source)
        layout.addWidget(self.combo_source)
        
        self.stacked = QStackedWidget()
        
        # Camera controls
        self.page_camera = QWidget()
        cam_layout = QHBoxLayout()
        self.btn_load_camera = QPushButton("Load Camera")
        self.lbl_camera_status = QLabel("")
        self.btn_load_camera.clicked.connect(self.load_camera)
        cam_layout.addWidget(self.btn_load_camera)
        cam_layout.addWidget(self.lbl_camera_status)
        self.page_camera.setLayout(cam_layout)
        self.stacked.addWidget(self.page_camera)
        
        # Media controls
        self.page_media = QWidget()
        media_layout = QHBoxLayout()
        self.btn_load_media = QPushButton("Load Media")
        self.lbl_media_status = QLabel("")
        self.btn_load_media.clicked.connect(self.load_media)
        media_layout.addWidget(self.btn_load_media)
        media_layout.addWidget(self.lbl_media_status)
        self.page_media.setLayout(media_layout)
        self.stacked.addWidget(self.page_media)
        
        # Network Stream controls
        self.page_stream = QWidget()
        stream_layout = QHBoxLayout()
        self.lineedit_url = QLineEdit()
        self.lineedit_url.setPlaceholderText("Enter stream URL")
        self.btn_load_stream = QPushButton("Load Stream")
        self.lbl_stream_status = QLabel("")
        self.btn_load_stream.clicked.connect(self.load_stream)
        stream_layout.addWidget(self.lineedit_url)
        stream_layout.addWidget(self.btn_load_stream)
        stream_layout.addWidget(self.lbl_stream_status)
        self.page_stream.setLayout(stream_layout)
        self.stacked.addWidget(self.page_stream)
        
        layout.addWidget(self.stacked)
        self.setLayout(layout)
    
    def change_source(self, index):
        self.stacked.setCurrentIndex(index)
        logging.info(f"Input source changed to index: {index}")
    
    def load_camera(self):
        self.lbl_camera_status.setText("Camera loaded")
        self.lbl_camera_status.setStyleSheet("color: green; font-size: 10px;")
        logging.info("Camera loaded")
    
    def load_media(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Media File", "",
                                                   "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All Files (*)")
        if file_path:
            self.mediaSelected.emit(file_path)
            filename = os.path.basename(file_path)
            self.lbl_media_status.setText(f"{filename} loaded")
            self.lbl_media_status.setStyleSheet("color: green; font-size: 10px;")
            logging.info(f"Media loaded: {filename}")
    
    def load_stream(self):
        url = self.lineedit_url.text().strip()
        if url:
            self.lbl_stream_status.setText("Stream loaded")
            self.lbl_stream_status.setStyleSheet("color: green; font-size: 10px;")
            logging.info(f"Stream loaded: {url}")
        else:
            self.lbl_stream_status.setText("Invalid URL")
            self.lbl_stream_status.setStyleSheet("color: red; font-size: 10px;")
            logging.info("Stream load failed: invalid URL")

class MainPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.media_file_mode = None
        self.frame_image = None
        self.prev_sup_labels = set()
        self.yolo_worker = YOLOPoseWorker()
        self.yolo_worker.peopleReady.connect(self.handlePeople)
        self.yolo_worker.start()
        self.last_entry_log_times = {}  # label -> timestamp
        self.last_exit_log_times  = {}
        self.entry_exit_log_path = "entry_exit.log"
        self.coords_log_path     = "coords.log"
        self.last_coords_log_time = time.time()
        self.motion_tracking_enabled = False
        self.people_enabled = False
        self.unsupervised_enabled = False
        self.supervised_enabled = False
        
        self.current_frame = None
        self.current_people_info = []
        self.current_unsup_info = []
        self.current_sup_info = []
        self.current_people_frame = None
        self.current_unsup_frame = None
        self.current_sup_frame = None

        self.initUI()
        self.initVideoProcessing()
    
    def initUI(self):
        self.video_label = QLabel("Video Display Area")
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; font-size: 32px; color: white;")
        
        self.btn_toggle_motion = QPushButton("Motion Tracking (OFF)")
        self.btn_toggle_people = QPushButton("People Recognition (OFF)")
        self.btn_toggle_unsupervised = QPushButton("Unsupervised Face Recognition (OFF)")
        self.btn_toggle_supervised = QPushButton("Supervised Face Recognition (OFF)")
        self.btn_exit = QPushButton("Exit")
        self.btn_settings = QPushButton()
        self.btn_settings.setFixedSize(40, 40)
        settings_icon = QIcon("settings.svg")
        self.btn_settings.setIcon(settings_icon)
        self.btn_settings.setIconSize(self.btn_settings.size())
        
        self.btn_toggle_motion.clicked.connect(self.toggle_motion)
        self.btn_toggle_people.clicked.connect(self.toggle_people)
        self.btn_toggle_unsupervised.clicked.connect(self.toggle_unsupervised)
        self.btn_toggle_supervised.clicked.connect(self.toggle_supervised)
        self.btn_exit.clicked.connect(self.close_app)
        self.btn_settings.clicked.connect(self.open_settings)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle_motion)
        button_layout.addWidget(self.btn_toggle_people)
        button_layout.addWidget(self.btn_toggle_unsupervised)
        button_layout.addWidget(self.btn_toggle_supervised)
        button_layout.addWidget(self.btn_settings)
        button_layout.addWidget(self.btn_exit)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, alignment=Qt.AlignHCenter)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        
    
    def initVideoProcessing(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.captureFrame)
        self.timer.start(30)
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        
        self.yolo_worker = YOLOPoseWorker()
        self.yolo_worker.peopleReady.connect(self.handlePeople)
        self.yolo_worker.start()
        
        self.unsupervised_worker = FaceRecognitionWorker(batch_size=4, cluster_interval=300, timeout=1.0)
        self.unsupervised_worker.recognitionReady.connect(self.handleUnsupervised)
        self.unsupervised_worker.start()
        
        self.supervised_worker = SupervisedFaceRecognitionWorker()
        self.supervised_worker.recognitionReady.connect(self.handleSupervised)
        self.supervised_worker.start()
    
    def setMediaSource(self, file_path):
        if self.cap.isOpened():
            self.cap.release()
        self.media_file_mode = None
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.frame_image = cv2.imread(file_path)
            self.media_file_mode = 'image'
        else:
            self.cap = cv2.VideoCapture(file_path)
            self.media_file_mode = 'video'
        logging.info(f"Media source set: {file_path}, mode: {self.media_file_mode}")
    
    def toggle_motion(self):
        self.motion_tracking_enabled = not self.motion_tracking_enabled
        self.btn_toggle_motion.setText(f"Motion Tracking ({'ON' if self.motion_tracking_enabled else 'OFF'})")
        logging.info(f"Motion Tracking turned {'ON' if self.motion_tracking_enabled else 'OFF'}")
    
    def toggle_people(self):
        self.people_enabled = not self.people_enabled
        self.btn_toggle_people.setText(f"People Recognition ({'ON' if self.people_enabled else 'OFF'})")
        logging.info(f"People Recognition turned {'ON' if self.people_enabled else 'OFF'}")
    
    def toggle_unsupervised(self):
        self.unsupervised_enabled = not self.unsupervised_enabled
        self.btn_toggle_unsupervised.setText(f"Unsupervised Face Recognition ({'ON' if self.unsupervised_enabled else 'OFF'})")
        logging.info(f"Unsupervised Face Recognition turned {'ON' if self.unsupervised_enabled else 'OFF'}")
    
    def toggle_supervised(self):
        self.supervised_enabled = not self.supervised_enabled
        self.btn_toggle_supervised.setText(f"Supervised Face Recognition ({'ON' if self.supervised_enabled else 'OFF'})")
        logging.info(f"Supervised Face Recognition turned {'ON' if self.supervised_enabled else 'OFF'}")
    
    def open_settings(self):
        self.stacked_widget.setCurrentIndex(1)
        logging.info("Navigated to Settings Panel")
    
    def close_app(self):
        logging.info("Exiting application from Main Page")
        self.yolo_worker.stop()
        self.unsupervised_worker.stop()
        self.supervised_worker.stop()
        if self.cap.isOpened():
            self.cap.release()
        QApplication.instance().quit()

    def captureFrame(self):
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
        self.current_frame = frame.copy()
        
        if self.motion_tracking_enabled:
            self.applyMotionTracking(self.current_frame)
        if self.people_enabled:
            self.yolo_worker.processFrame(frame)
        if self.unsupervised_enabled:
            self.unsupervised_worker.processFaces(frame, [])
        if self.supervised_enabled:
            self.supervised_worker.processFrame(frame)
        
        self.updateCompositeDisplay()
    
    def applyMotionTracking(self, frame):
        mask = self.bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    def updateCompositeDisplay(self):
        frame_to_show = None
        source_info = "Unknown"

        if self.supervised_enabled and self.current_sup_frame is not None:
            source_info = "Supervised"
            frame_to_show = self.current_sup_frame.copy()
            info = self.current_sup_info
            color = (0, 0, 255); text_prefix = ""
            for (x1, y1, x2, y2), label in info:
                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_to_show, f"{text_prefix}{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif self.unsupervised_enabled and self.current_unsup_frame is not None:
            source_info = "Unsupervised"
            frame_to_show = self.current_unsup_frame.copy()
            info = self.current_unsup_info
            color = (255, 0, 0); text_prefix = "ID "
            for (x1, y1, x2, y2), label in info:
                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_to_show, f"{text_prefix}{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif self.people_enabled and self.current_people_frame is not None:
            source_info = "People (Annotated)"
            frame_to_show = self.current_people_frame.copy()

        elif self.motion_tracking_enabled and self.current_frame is not None:
            source_info = "Motion (Applied to Raw)"
            frame_to_show = self.current_frame.copy()
            self.applyMotionTracking(frame_to_show)

        else:
            if self.current_frame is not None:
                source_info = "Raw Fallback"
                frame_to_show = self.current_frame.copy()
            else:
                source_info = "None"
                frame_to_show = np.zeros((720, 1280, 3), dtype=np.uint8)

        logging.debug(f"UpdateDisplay selected source: {source_info}") 

        if frame_to_show is not None and frame_to_show.shape[0] > 0 and frame_to_show.shape[1] > 0:
            if frame_to_show.shape[0] != 720 or frame_to_show.shape[1] != 1280:
                logging.warning(f"Frame from '{source_info}' has unexpected shape: {frame_to_show.shape}. Resizing.")
                try:
                    frame_to_show = cv2.resize(frame_to_show, (1280, 720))
                except cv2.error as e:
                    logging.error(f"Failed to resize frame from '{source_info}': {e}")
                    self.video_label.setText("Resize Error")
                    self.video_label.setStyleSheet("background-color: red; color: white;")
                    return 

            logging.debug(f"Attempting to display frame from '{source_info}' with final shape {frame_to_show.shape}") 
            try:
                h, w, ch = frame_to_show.shape
                if ch == 3:
                    bytes_per_line = ch * w
                    if not frame_to_show.flags['C_CONTIGUOUS']:
                        frame_to_show = np.ascontiguousarray(frame_to_show)

                    qimg = QImage(frame_to_show.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    self.video_label.setPixmap(QPixmap.fromImage(qimg))
                else:
                    logging.error(f"Frame from '{source_info}' has incorrect channels: {ch}")
                    self.video_label.setText(f"Bad Frame Channels: {ch}")
                    self.video_label.setStyleSheet("background-color: red; color: white;")
            except Exception as e:
                logging.error(f"Error converting/displaying frame from '{source_info}': {e}")
                self.video_label.setText("Error Displaying Frame")
                self.video_label.setStyleSheet("background-color: red; color: white;")
        else:
            logging.warning(f"No valid frame_to_show for display (Source attempted: {source_info}).")
            self.video_label.setText("No Frame Available")
            self.video_label.setStyleSheet("background-color: black; color: gray;")
        
    def handlePeople(self, frame, people_info):
        if frame is not None:
            logging.debug(f"handlePeople received annotated frame (Shape: {frame.shape})")
            self.current_people_frame = frame 
            self.current_people_info = people_info
        else:
            logging.warning("handlePeople received None frame.")

    def handleUnsupervised(self, frame, face_info):
        self.current_unsup_frame = frame
        self.current_unsup_info = face_info
        self.updateCompositeDisplay()
    
    def handleSupervised(self, frame, face_info):
        self.current_sup_frame = frame
        self.current_sup_info  = face_info

        now = time.time()
        current_labels = { label for (_, label) in face_info }

        for label in current_labels - self.prev_sup_labels:
            last_ent = self.last_entry_log_times.get(label, 0)
            if now - last_ent >= 3600:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = f"{label} entered at {ts}"
                with open(self.entry_exit_log_path, "a") as f:
                    f.write(msg + "\n")
                logging.info(msg)
                self.last_entry_log_times[label] = now

        for label in self.prev_sup_labels - current_labels:
            last_ex = self.last_exit_log_times.get(label, 0)
            if now - last_ex >= 3600:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = f"{label} exited at {ts}"
                with open(self.entry_exit_log_path, "a") as f:
                    f.write(msg + "\n")
                logging.info(msg)
                self.last_exit_log_times[label] = now

        self.prev_sup_labels = current_labels

        if now - self.last_coords_log_time >= 3.0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for (x1, y1, x2, y2), label in face_info:
                coord_msg = f"{label} at ({x1},{y1},{x2},{y2}) at {ts}"
                with open(self.coords_log_path, "a") as f:
                    f.write(coord_msg + "\n")
                logging.info(coord_msg)
            self.last_coords_log_time = now

        self.updateCompositeDisplay()

class SettingsPanel(QWidget):    
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        full_cfg = load_app_config()
        smtp_cfg = full_cfg.get("smtp", {})
        self.smtp_host     = smtp_cfg.get("host", "")
        self.smtp_port     = smtp_cfg.get("port",  587)
        self.smtp_user     = smtp_cfg.get("username", "")
        self.smtp_pass     = smtp_cfg.get("password", "")
        self.smtp_use_tls  = smtp_cfg.get("use_tls", True)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        model_group = QGroupBox("Model Paths")
        model_layout = QVBoxLayout()
        self.model_row1 = ModelPathRow("YOLO Pose Model:")
        self.model_row2 = ModelPathRow("Supervised Classifier (.pkl):")
        model_layout.addWidget(self.model_row1)
        model_layout.addWidget(self.model_row2)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        self.input_source_panel = InputSourcePanel()
        input_layout.addWidget(self.input_source_panel)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        smtp_group = QGroupBox("SMTP Settings")
        smtp_form  = QFormLayout()

        self.edit_smtp_host = QLineEdit(self.smtp_host)
        smtp_form.addRow("SMTP Host:", self.edit_smtp_host)

        self.edit_smtp_port = QLineEdit(str(self.smtp_port))
        smtp_form.addRow("SMTP Port:", self.edit_smtp_port)

        self.edit_smtp_user = QLineEdit(self.smtp_user)
        smtp_form.addRow("Username:", self.edit_smtp_user)

        self.edit_smtp_pass = QLineEdit(self.smtp_pass)
        self.edit_smtp_pass.setEchoMode(QLineEdit.Password)
        smtp_form.addRow("Password:", self.edit_smtp_pass)

        self.chk_smtp_tls = QCheckBox("Use TLS")
        self.chk_smtp_tls.setChecked(self.smtp_use_tls)
        smtp_form.addRow(self.chk_smtp_tls)

        btn_save_smtp = QPushButton("Save SMTP Settings")
        btn_save_smtp.clicked.connect(self.save_smtp_settings)
        smtp_form.addRow(btn_save_smtp)

        smtp_group.setLayout(smtp_form)
        layout.addWidget(smtp_group)

        self.btn_theme_panel = QPushButton("Theme Settings")
        self.btn_theme_panel.clicked.connect(self.open_theme_panel)
        layout.addWidget(self.btn_theme_panel)

        bottom_layout = QHBoxLayout()
        self.btn_back_home = QPushButton("Back to Home")
        self.btn_back_home.clicked.connect(self.back_to_home)
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.exit_app)
        bottom_layout.addWidget(self.btn_back_home)
        bottom_layout.addWidget(self.btn_exit)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)
        logging.info("Settings Panel initialized")

    def save_smtp_settings(self):
        cfg = load_app_config()
        try:
            port = int(self.edit_smtp_port.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Port", "SMTP Port must be an integer.")
            return
        cfg["smtp"] = {
            "host":     self.edit_smtp_host.text().strip(),
            "port":     port,
            "username": self.edit_smtp_user.text().strip(),
            "password": self.edit_smtp_pass.text(),
            "use_tls":  self.chk_smtp_tls.isChecked()
        }
        save_app_config(cfg)
        QMessageBox.information(self, "Saved", "SMTP settings saved successfully.")
        logging.info("SMTP settings updated")    

    def open_theme_panel(self):
        self.stacked_widget.setCurrentIndex(2)
        logging.info("Navigated to Theme Settings")
    
    def back_to_home(self):
        self.stacked_widget.setCurrentIndex(0)
        logging.info("Navigated back to Home")
    
    def exit_app(self):
        logging.info("Exiting application from Settings Panel")
        QApplication.instance().quit()

class ThemePanel(QWidget):
    def __init__(self, stacked_widget):
        super().__init__(stacked_widget)
        self.stacked_widget = stacked_widget
        full_cfg  = load_app_config()
        theme_cfg = full_cfg.get("theme", {})
        self.current_theme           = theme_cfg.get("mode",       "Dark")
        self.custom_background_color = theme_cfg.get("background", "#2B2B2B")
        self.custom_button_color     = theme_cfg.get("button",     "#444444")
        self.custom_text_color       = theme_cfg.get("text",       "#FFFFFF")
        self.initUI()
        self.apply_theme()
        self.update_button_text()
    
    def initUI(self):
        layout = QVBoxLayout()
        title = QLabel("Theme Settings")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        h_layout = QHBoxLayout()
        theme_group = QGroupBox("Theme")
        theme_group_layout = QVBoxLayout()
        self.btn_toggle_theme = QPushButton("")
        self.btn_toggle_theme.clicked.connect(self.toggle_theme)
        theme_group_layout.addWidget(self.btn_toggle_theme)
        theme_group.setLayout(theme_group_layout)
        h_layout.addWidget(theme_group)
        
        custom_group = QGroupBox("Custom Theme Settings")
        custom_layout = QVBoxLayout()
        bg_layout = QHBoxLayout()
        self.btn_pick_bg = QPushButton("Pick Background Color")
        self.btn_pick_bg.clicked.connect(self.pick_background_color)
        bg_layout.addWidget(self.btn_pick_bg)
        self.bg_preview = QFrame()
        self.bg_preview.setFixedSize(30, 30)
        self.bg_preview.setStyleSheet(f"background-color: {self.custom_background_color}; border: 1px solid black;")
        bg_layout.addWidget(self.bg_preview)
        custom_layout.addLayout(bg_layout)
        
        btn_layout = QHBoxLayout()
        self.btn_pick_button = QPushButton("Pick Button Color")
        self.btn_pick_button.clicked.connect(self.pick_button_color)
        btn_layout.addWidget(self.btn_pick_button)
        self.btn_preview = QFrame()
        self.btn_preview.setFixedSize(30, 30)
        self.btn_preview.setStyleSheet(f"background-color: {self.custom_button_color}; border: 1px solid black;")
        btn_layout.addWidget(self.btn_preview)
        custom_layout.addLayout(btn_layout)
        
        text_layout = QHBoxLayout()
        self.btn_pick_text = QPushButton("Pick Text Color")
        self.btn_pick_text.clicked.connect(self.pick_text_color)
        text_layout.addWidget(self.btn_pick_text)
        self.text_preview = QFrame()
        self.text_preview.setFixedSize(30, 30)
        self.text_preview.setStyleSheet(f"background-color: {self.custom_text_color}; border: 1px solid black;")
        text_layout.addWidget(self.text_preview)
        custom_layout.addLayout(text_layout)
        
        self.btn_apply_custom = QPushButton("Apply Custom Theme")
        self.btn_apply_custom.clicked.connect(self.apply_custom_theme)
        custom_layout.addWidget(self.btn_apply_custom)
        custom_group.setLayout(custom_layout)
        h_layout.addWidget(custom_group)
        
        layout.addLayout(h_layout)
        
        bottom_layout = QHBoxLayout()
        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(lambda: QApplication.instance().quit())
        bottom_layout.addWidget(self.btn_home)
        bottom_layout.addWidget(self.btn_exit)
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)
        self.update_button_text()
    
    def update_button_text(self):
        if self.current_theme == "Dark":
            self.btn_toggle_theme.setText("Switch to Light Mode")
        elif self.current_theme == "Light":
            self.btn_toggle_theme.setText("Switch to Dark Mode")
        else:
            self.btn_toggle_theme.setText("Switch to Standard Mode")
    
    def toggle_theme(self):
        if self.current_theme == "Dark":
            self.custom_background_color = "#FFFFFF"
            self.custom_button_color = "#ddd"
            self.custom_text_color = "#000000"
            self.current_theme = "Light"
            logging.info("Theme changed to Light")
        else:
            self.custom_background_color = "#2B2B2B"
            self.custom_button_color = "#444444"
            self.custom_text_color = "#FFFFFF"
            self.current_theme = "Dark"
            logging.info("Theme changed to Dark")
        
        self.apply_theme()
        self.update_button_text()
        self.update_previews()
    
    def pick_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.custom_background_color = color.name()
            logging.info(f"Custom background color set to {self.custom_background_color}")
            self.update_previews()
    
    def pick_button_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.custom_button_color = color.name()
            logging.info(f"Custom button color set to {self.custom_button_color}")
            self.update_previews()
    
    def pick_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.custom_text_color = color.name()
            logging.info(f"Custom text color set to {self.custom_text_color}")
            self.update_previews()
    
    def apply_custom_theme(self):
        self.apply_theme()
        logging.info("Custom theme applied")
        self.update_previews()
    
    def update_previews(self):
        self.bg_preview.setStyleSheet(f"background-color: {self.custom_background_color}; border: 1px solid black;")
        self.btn_preview.setStyleSheet(f"background-color: {self.custom_button_color}; border: 1px solid black;")
        self.text_preview.setStyleSheet(f"background-color: {self.custom_text_color}; border: 1px solid black;")
        logging.info("Color previews updated")
    
    def apply_theme(self):
        style = f"""
        QMainWindow {{ background-color: {self.custom_background_color}; color: {self.custom_text_color}; }}
        QLabel       {{ color: {self.custom_text_color}; }}
        QPushButton  {{
            background-color: {self.custom_button_color};
            color: {self.custom_text_color};
            padding: 10px;
            font-size: 14px;
            border: none;
        }}
        QGroupBox    {{ font-size: 16px; color: {self.custom_text_color}; }}
        QComboBox    {{ background-color: {self.custom_button_color}; color: {self.custom_text_color}; padding: 5px; }}
        """
        QApplication.instance().setStyleSheet(style)
        full_cfg = load_app_config()
        full_cfg["theme"] = {
            "mode":       self.current_theme,
            "background": self.custom_background_color,
            "button":     self.custom_button_color,
            "text":       self.custom_text_color
        }
        save_app_config(full_cfg)
        logging.info("Theme applied and configuration saved")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Multi-Feature Recognition System")
        self.setGeometry(100, 100, 1280, 720)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.main_page = MainPage(self.stacked_widget)         
        self.settings_panel = SettingsPanel(self.stacked_widget)  
        self.theme_panel = ThemePanel(self.stacked_widget)   

        self.settings_panel.input_source_panel.mediaSelected.connect(self.main_page.setMediaSource)
        self.settings_panel.model_row1.modelPathLoaded.connect(self.main_page.yolo_worker.setModelPath)
        self.settings_panel.model_row2.modelPathLoaded.connect(self.main_page.supervised_worker.loadClassifier)
        
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.settings_panel)
        self.stacked_widget.addWidget(self.theme_panel)
        logging.info("Main window initialized")
        export_menu = self.menuBar().addMenu("Export Logs")
        for fmt, label in [("html", "As HTML…"), ("csv", "As CSV…"), ("json", "As JSON…")]:
            act = QAction(label, self)
            act.triggered.connect(lambda _, f=fmt: self.export_logs(f))
            export_menu.addAction(act)
    
    def export_logs(self, fmt: str):
        filters = {
            "html": "HTML Files (*.html)",
            "csv":  "CSV Files (*.csv)",
            "json": "JSON Files (*.json)"
        }
        default_name = f"logs.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self,
            f"Save logs as {fmt.upper()}",
            default_name,
            filters[fmt]
        )
        if not path:
            return
        logs = self._read_all_logs()
        if fmt == "csv":
            self._write_csv(logs, path)
        elif fmt == "json":
            self._write_json(logs, path)
        else:
            self._write_html(logs, path)
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Export Complete")
        dlg.setText(f"Logs saved to:\n{path}")
        chk = QCheckBox("Send by email", dlg)
        dlg.setCheckBox(chk)
        dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if dlg.exec_() == QMessageBox.Ok and chk.isChecked():
            recipient, ok = QInputDialog.getText(self, "Send Logs", "Recipient email address:")
            if ok and recipient:
                try:
                    self._send_logs_email(path, recipient)
                    QMessageBox.information(self, "Email Sent", f"Logs emailed to {recipient}")
                except Exception as e:
                    QMessageBox.critical(self, "Email Failed", f"Failed to send logs: {e}")
    
    def _read_all_logs(self):
        entries = []
        for fname, src in [
            ("entry_exit.log", "entry_exit"),
            ("coords.log",     "coords"),
          #  ("app.log",        "app")
        ]:
            if not os.path.exists(fname):
                continue
            with open(fname) as f:
                for line in f:
                    entries.append({"source": src, "raw_line": line.strip()})
        return entries
    
    def _write_csv(self, logs, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "raw_line"])
            w.writeheader()
            w.writerows(logs)
    
    def _write_json(self, logs, path):
        import json
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
    
    def _write_html(self, logs, path):
        with open(path, "w") as f:
            f.write("<html><head><title>Logs Export</title></head><body>\n")
            f.write("<table border='1'><tr><th>Source</th><th>Entry</th></tr>\n")
            for e in logs:
                f.write(f"<tr><td>{e['source']}</td><td>{e['raw_line']}</td></tr>\n")
            f.write("</table></body></html>")
    
    def _send_logs_email(self, filepath: str, recipient: str):
        cfg = load_app_config().get("smtp", {})
        if not cfg:
            raise RuntimeError("SMTP settings not found in application_config.json")
        msg = EmailMessage()
        msg["From"]    = cfg["username"]
        msg["To"]      = recipient
        msg["Subject"] = f"Log export: {os.path.basename(filepath)}"
        msg.set_content("Please find the attached logs.")
        with open(filepath, "rb") as f:
            data = f.read()
        msg.add_attachment(data,
                           maintype="application",
                           subtype="octet-stream",
                           filename=os.path.basename(filepath))
        try:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SMTP server: {e}")
        try:
            if cfg.get("use_tls", False):
                server.starttls()
        except Exception as e:
            server.quit()
            raise RuntimeError(f"Failed to start TLS: {e}")
        try:
            server.login(cfg["username"], cfg["password"])
        except Exception as e:
            server.quit()
            raise RuntimeError(f"SMTP login failed: {e}")
        try:
            server.send_message(msg)
        except Exception as e:
            server.quit()
            raise RuntimeError(f"Failed to send email: {e}")
        server.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    if not load_app_config():
        dark_style = """
        QMainWindow { background-color: #2B2B2B; color: white; }
        QLabel { color: white; }
        QPushButton {
            background-color: #444444;
            color: white;
            padding: 10px;
            font-size: 14px;
            border: none;
        }
        QGroupBox { font-size: 16px; color: #FFD700; }
        QComboBox { background-color: #444444; color: white; padding: 5px; }
        """
        app.setStyleSheet(dark_style)
        logging.info("Default dark style applied at startup")
    
    window = MainWindow()
    window.show()
    logging.info("Application started")
    sys.exit(app.exec_())
