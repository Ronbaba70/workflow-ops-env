FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install pydantic openai fastapi uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
