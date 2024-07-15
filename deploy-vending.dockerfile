FROM python:3.9-slim

WORKDIR /code

RUN apt-get update
RUN apt-get install -y openssl
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /code/requirements.txt
COPY ./Vending-YOLOv8n.onnx /code/Vending-YOLOv8n.onnx
COPY ./vending_app.py /code/vending_app.py

RUN pip3 install -r /code/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/code/vending_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
