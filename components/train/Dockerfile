FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY components/train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY components/train/train.py .
COPY data/ /app/data/

ENTRYPOINT ["python", "train.py"]