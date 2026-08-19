# CCTV Face Recognition System

A machine learning-based CCTV face recognition and analysis system that explores both **supervised and unsupervised learning approaches** for identifying and grouping faces from video streams.

The project uses **InsightFace** for facial feature extraction, **Support Vector Machine (SVM)** for supervised face classification, and **DBSCAN** for unsupervised face clustering.

## 📌 Project Overview

Traditional CCTV systems require continuous human monitoring to identify individuals appearing in surveillance footage. This project explores how computer vision and machine learning can be used to automatically detect, analyze, classify, and group faces captured from CCTV/video sources.

Two different machine learning approaches are explored:

### Supervised Learning

The supervised approach uses labeled facial images to train a classifier.

The general workflow is:

```text
Labeled Face Images
        ↓
Face Detection
        ↓
InsightFace
        ↓
512-D Face Embeddings
        ↓
SVM Training
        ↓
Trained Face Classifier
        ↓
Face Recognition
```

A **Support Vector Machine (SVM)** classifier is trained using facial embeddings extracted from labeled images.

### Unsupervised Learning

The unsupervised approach attempts to group detected faces without requiring predefined identity labels.

```text
Detected Faces
      ↓
InsightFace
      ↓
Face Embeddings
      ↓
DBSCAN Clustering
      ↓
Face Groups / Clusters
```

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is used to group similar facial embeddings while allowing unidentified/outlier faces to be treated as noise.

---

## ✨ Key Features

* Face detection and analysis from video sources
* Facial feature extraction using InsightFace
* 512-dimensional facial embeddings
* Supervised face classification using SVM
* Unsupervised face clustering using DBSCAN
* CCTV/video processing using OpenCV
* Graphical interface using PyQt5
* Local SQLite database support
* Configurable application settings
* Support for machine-learning-based face analysis

---

## 🧠 Machine Learning Approaches

| Approach              | Technique   | Purpose                                                     |
| --------------------- | ----------- | ----------------------------------------------------------- |
| Feature Extraction    | InsightFace | Generate numerical representations of detected faces        |
| Supervised Learning   | SVM         | Classify faces using previously labeled identities          |
| Unsupervised Learning | DBSCAN      | Automatically group similar faces without predefined labels |

### Why use both approaches?

The supervised approach is useful when the identities of individuals are already known and labeled training images are available.

The unsupervised approach is useful when identities are not known beforehand. Instead of requiring predefined labels, the system can group visually similar face embeddings into clusters.

This allows the project to explore the differences between **identity-based classification** and **automatic face grouping**.

---

## 🛠️ Technologies Used

| Technology   | Purpose                                  |
| ------------ | ---------------------------------------- |
| Python       | Main programming language                |
| OpenCV       | Video and image processing               |
| InsightFace  | Face detection and facial embeddings     |
| Scikit-learn | SVM classification and DBSCAN clustering |
| NumPy        | Numerical operations                     |
| PyQt5        | Graphical user interface                 |
| SQLite       | Local application data storage           |
| ONNX Runtime | Model inference                          |
| Ultralytics  | Computer vision/model functionality      |

---

## 📁 Project Structure

```text
miniproj/
│
├── cctv.py
│   └── Main CCTV face recognition application
│
├── classifiertraining.py
│   └── Supervised SVM classifier training
│
├── unsup.py
│   └── Unsupervised DBSCAN-based face clustering
│
├── requirements.txt
│   └── Python dependencies
│
├── application_config.example.json
│   └── Example application configuration
│
├── .gitignore
│   └── Files excluded from version control
│
└── README.md
    └── Project documentation
```

Runtime-generated files such as the SQLite database and application logs are excluded from version control.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AbhiramSajeev405/miniproj.git
cd miniproj
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

#### Windows

```powershell
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

The repository provides:

```text
application_config.example.json
```

Create a local configuration file before running the application.

### Windows PowerShell

```powershell
Copy-Item application_config.example.json application_config.json
```

### Linux/macOS

```bash
cp application_config.example.json application_config.json
```

Update the local `application_config.json` with the appropriate configuration values.

> `application_config.json` is excluded from Git to prevent local configuration or credentials from being accidentally committed.

---

## ▶️ Running the Project

### Main CCTV Application

```bash
python cctv.py
```

### Unsupervised Face Clustering

```bash
python unsup.py
```

### Supervised Classifier Training

The supervised classifier training logic is implemented in:

```text
classifiertraining.py
```

The training process extracts facial embeddings from labelled images and trains an SVM classifier using those embeddings.

---

## 🔬 Supervised vs. Unsupervised Approach

| Supervised                     | Unsupervised                               |
| ------------------------------ | ------------------------------------------ |
| Requires labelled identities    | Does not require predefined identities     |
| Uses SVM                       | Uses DBSCAN                                |
| Learns known face classes      | Discovers groups automatically             |
| Suitable for known individuals | Suitable for unknown/unlabelled individuals |
| Classification problem         | Clustering problem                         |

---

## 🔒 Privacy Considerations

Face recognition systems process biometric information and should be deployed responsibly.

For this reason, runtime databases, local configuration files, and generated application data are not intended to be stored directly in the public repository.

Any real-world deployment should consider:

* User consent
* Data protection requirements
* Secure storage of facial embeddings
* Access control
* Data retention policies
* Applicable privacy laws and regulations


---

## ⚠️ Disclaimer

This project was developed for **academic and educational purposes**. Face recognition technologies should be used responsibly and in accordance with applicable privacy, security, and data-protection requirements.

---
## 📚 Project Context

The **CCTV Face Recognition System** was developed collaboratively by our four-member academic project team as part of our academic work. Our team worked together on the design, development, implementation, testing, and documentation of the system.

The project focuses on applying **computer vision and face recognition techniques** to CCTV-based surveillance, with the goal of detecting and recognizing individuals from video footage and exploring the practical application of machine learning in intelligent surveillance systems.

The development of this project was a collaborative effort by **Abhiram Sajeev, Adarsh S J, Alfin Jerome, and Alen J S**.

## 👥 Project Team

This project was designed and developed collaboratively by our four-member academic project team. We worked together on the development, testing, and documentation of the CCTV Face Recognition System.

| Team Member | GitHub |
|-------------|--------|
| **Abhiram Sajeev** | [@AbhiramSajeev405](https://github.com/AbhiramSajeev405) |
| **Adarsh S J** | [@Horcrux123](https://github.com/Horcrux123) |
| **Alfin Jerome** | [@alfinjerome](https://github.com/alfinjerome) |
| **Alen J S** | [@thereelalen](https://github.com/thereelalen) |

> This repository represents the collaborative work of all four team members.



