## Project Structure

```bash
apple-detector-pipeline/
├── common/
│   ├── dataset.py        # Lớp VOCDataset (đọc ảnh + xml)
│   └── model_utils.py    # Hàm tạo mô hình Faster R-CNN
├── components/
│   ├── train/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── train.py
│   ├── evaluate/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── evaluate.py
│   └── serve/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── app.py        # Flask API
│       └── model_loader.py
├── pipeline/
│   └── pipeline.py       # Định nghĩa workflow Kubeflow
├── data/
│   ├── train/            # images + xml
│   ├── valid/
│   └── test/
└── .github/workflows/    # (tuỳ chọn) CI/CD
```
