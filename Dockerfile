# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model, scaler, and the rest of the application code
COPY wine_quality_model.pkl .
COPY scaler.pkl .
COPY . .

# Expose Flask app port (default is 5000)
EXPOSE 5000

# Define the command to run the app
ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

