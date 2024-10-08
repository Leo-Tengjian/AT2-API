FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app/frontend /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
