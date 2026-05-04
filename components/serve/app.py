%%writefile components/serve/app.py
import sys
import os
sys.path.insert(0, '/content/Apple-detector-pipeline-MLOps')
sys.path.insert(0, '/content/Apple-detector-pipeline-MLOps/components/serve')  # Thêm dòng này

import io
import argparse
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as T
import model_loader

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    transform = T.ToTensor()
    img_tensor = transform(img)
    result = model_loader.predict(img_tensor)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    model_loader.load_model(args.model_path)
    app.run(host='0.0.0.0', port=args.port)