FROM continuumio/miniconda3

WORKDIR /home/app

# Update package lists and install dependencies
RUN apt-get update -y 
RUN apt-get install nano unzip
RUN apt-get install -y python3.10
RUN apt install curl -y

# Install Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Install Python dependencies
COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# Copy the application code
COPY . /home/app

# Run the API with Gunicorn and Uvicorn
CMD gunicorn api:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker 