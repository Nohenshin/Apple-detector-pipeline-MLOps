apple-detector-pipeline/
├── common/
│   ├── __init__.py
│   ├── dataset.py
│   └── model_utils.py
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
│       ├── app.py
│       └── model_loader.py
├── pipeline/
│   └── pipeline.py
├── data/                 
├── .gitignore
└── README.md