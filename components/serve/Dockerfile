FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY components/serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY components/serve/model_loader.py components/serve/app.py ./

ENTRYPOINT ["python", "app.py"]