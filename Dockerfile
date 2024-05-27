FROM python:3.11.5
RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
RUN 

CMD ["python3", "app.py"]