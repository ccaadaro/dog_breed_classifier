# 🐶 Dog Breed Classifier — Full ML Deployment Pipeline

This project implements a complete machine learning deployment workflow using the Dog Breed Identification dataset from Kaggle.

It includes:

- 🏋️ **Model Training**: Train a fine-tuned ResNet18 on high-resolution RGB images of dogs
- 🌐 **REST API**: Flask endpoint that serves predictions for uploaded dog images
- 💻 **Web App**: HTML form to upload an image and get predicted dog breed with confidence
- 📊 **Monitoring**: Prometheus + Grafana dashboard for live inference monitoring
- ☁️ **Cloud Ready**: Dockerized for deployment on Render, AWS, etc.

---

## 🔧 Tech Stack

- PyTorch  
- Flask  
- HTML/CSS  
- Prometheus & Grafana  
- Docker  
- Python 3.10  
- Kaggle CLI for dataset access

---

## 🚀 Quick Start

### 1. Setup Kaggle API

Make sure you have your `kaggle.json` API key (from your Kaggle account settings).
Place it at:
```bash
~/.kaggle/kaggle.json
```

### 2. Download and Prepare Dataset
```bash
cd training
python download_kaggle_data.py
```

### 3. Train the Model
```bash
python train.py
```

### 4. Launch Flask API
```bash
cd ../api
python app.py
```

### 5. Launch Web App
```bash
cd ../webapp
python app.py
```

### 6. Open Browser
Visit `http://localhost:8000` to upload an image and view predictions.

---

This project is perfect for demonstrating a real-world ML application, from training with real images to serving and monitoring models in production.

Contributions welcome!
