from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import pandas as pd
import os
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Load label map
labels_csv = '../training/data/dog-breed-identification/labels.csv'
breeds = sorted(pd.read_csv(labels_csv)['breed'].unique())

# Load model
model = get_model(num_classes=len(breeds))
model.load_state_dict(torch.load("resnet18_dogbreed.pt", map_location=torch.device('cpu')))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()

    topk = torch.topk(probs, k=5)
    top_probs = topk.values.tolist()
    top_indices = topk.indices.tolist()
    top_labels = [breeds[i] for i in top_indices]

    predictions = [{"breed": label, "confidence": float(prob)} for label, prob in zip(top_labels, top_probs)]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

