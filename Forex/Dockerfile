FROM python:3.11-slim
WORKDIR /app
# Copy requirements first so Docker can cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# Copy the application, data and models
COPY . .
# Streamlit's default port
EXPOSE 8501
# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]