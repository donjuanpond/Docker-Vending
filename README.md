# Docker-Vending
Vending machine item detection in a docker container. Running it creates a Streamlit app to interface with the YOLOv8n model.

To pull the docker container image, run `docker pull donjuanpond/vending-streamlit`. Alternatively, if for some reason you want to build it on your end, you can download the zip of the repo and, upon `cd`ing inside the folder, run `docker build -t vending-streamlit -f deploy-vending.dockerfile .`.

To run the container without webcam access, do `docker run -p 8501:8501 vending-streamlit` and visit `localhost:8501` to see your Streamlit app in action. If you want to give it webcam access, do the following:
- On Linux, just do `docker run -p 8501:8501 --privileged -v /dev/video0:/dev/video0 vending-streamlit`
- On Windows, good luck, send me a message if you figure it out
