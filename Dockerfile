# Use an official Python runtime as the parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable for OpenCV to run in headless mode
ENV OPENCV_HEADLESS=1

# Run the Python script when the container launches
CMD ["python", "./main_v8.py"]
