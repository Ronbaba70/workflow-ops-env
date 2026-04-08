FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install pydantic openai fastapi uvicorn
CMD ["python", "inference.py"]
