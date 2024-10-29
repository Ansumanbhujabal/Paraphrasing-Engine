# Use the official Python image from the Docker Hub
FROM python:3.10.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the NLTK data directory and download required resources
RUN mkdir -p /usr/local/share/nltk_data && \
    python3 -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt punkt_tab

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
