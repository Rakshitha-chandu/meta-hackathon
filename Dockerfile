FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]