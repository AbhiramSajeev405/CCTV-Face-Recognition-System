import sys
import os
import cv2
import numpy as np
import insightface
import torch
import pickle
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout
from sklearn.svm import SVC

# ---------------------------
# TrainingWorker: A QThread to perform the supervised training process.
# ---------------------------
class TrainingWorker(QThread):
    # Signal to notify when training is done. Emits a message string.
    trainingDone = pyqtSignal(str)
    
    def __init__(self, train_folder, parent=None):
        super().__init__(parent)
        self.train_folder = train_folder
        self.running = True
        
        # Initialize InsightFace FaceAnalysis for embedding extraction.
        self.face_app = insightface.app.FaceAnalysis(name='buffalo_l')
        # Using ctx_id=0 to use GPU if available.
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    def run(self):
        # Lists to accumulate embeddings and labels
        embeddings = []
        labels = []
        
        # Assume train_folder contains subfolders, each named as the identity
        for person_name in os.listdir(self.train_folder):
            person_folder = os.path.join(self.train_folder, person_name)
            if not os.path.isdir(person_folder):
                continue
            print(f"Processing images for {person_name}...")
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = self.face_app.get(rgb_img)
                    # If at least one face is detected, use the first one
                    if faces:
                        emb = faces[0].embedding  # 512-d vector
                        embeddings.append(emb)
                        labels.append(person_name)
                        print(f"Processed {filename} for {person_name}")
        if embeddings and labels:
            embeddings_array = np.array(embeddings)
            # Train an SVM classifier on the embeddings
            print("Training classifier...")
            classifier = SVC(probability=True, kernel='linear')
            classifier.fit(embeddings_array, labels)
            # Save the trained classifier to disk
            with open("classifier.pkl", "wb") as f:
                pickle.dump(classifier, f)
            self.trainingDone.emit("Training complete! Classifier saved as classifier.pkl.")
        else:
            self.trainingDone.emit("No valid face data found. Training aborted.")

# ---------------------------
# MainWindow: Provides a simple UI to select the training folder and start training.
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Supervised Face Recognition Training")
        self.setGeometry(100, 100, 600, 300)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        
        # Label to show status
        self.status_label = QLabel("Select a training folder to start.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        # Button to select training folder
        self.btn_select_folder = QPushButton("Select Training Folder")
        self.btn_select_folder.clicked.connect(self.selectFolder)
        self.layout.addWidget(self.btn_select_folder)
        
        # Button to start training
        self.btn_train = QPushButton("Train Model")
        self.btn_train.clicked.connect(self.trainModel)
        self.btn_train.setEnabled(False)  # Disabled until a folder is selected
        self.layout.addWidget(self.btn_train)
        
        self.central_widget.setLayout(self.layout)
        
        self.train_folder = None
        self.training_worker = None
    
    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Training Folder", "")
        if folder:
            self.train_folder = folder
            self.status_label.setText(f"Selected folder: {folder}")
            self.btn_train.setEnabled(True)
    
    def trainModel(self):
        if self.train_folder is None:
            self.status_label.setText("No training folder selected!")
            return
        self.status_label.setText("Training in progress...")
        # Create and start the training worker thread
        self.training_worker = TrainingWorker(self.train_folder)
        self.training_worker.trainingDone.connect(self.onTrainingDone)
        self.training_worker.start()
    
    def onTrainingDone(self, message):
        self.status_label.setText(message)
        self.btn_train.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
