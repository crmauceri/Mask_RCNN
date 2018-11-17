# Use an official tensorflow runtime as a parent image

# For CPU, use
FROM tensorflow/tensorflow

# For GPU use 
# FROM tensorflow/tensorflow:latest-gpu

# Set the working directory to /app
WORKDIR /app

# Copy the Mask_RCNN directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python setup.py install
    

# Compile cocoapi 
WORKDIR /app/coco/PythonAPI/
RUN make

# Define environment variable
ENV NAME MaskRCNN

