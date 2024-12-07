# Use the official Python 3.12 image as a base
FROM python:3.12-slim
#
## Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
## Set working directory
WORKDIR /code
#
## Install dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v
#
## Copy the current directory contents into the container at /code
COPY . /code/
#
## Expose the port the app runs on
EXPOSE 5000
#
## Activate the virtual environment and run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]