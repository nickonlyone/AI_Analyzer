FROM python:3.10.18

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git build-essential

RUN pip install torch torchvision

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p audio reports etalons static templates

EXPOSE 8000

WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]