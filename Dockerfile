# Use an official tensorflow runtime as a parent image

# For CPU, use
FROM tensorflow/tensorflow:latest-py3

# For GPU use 
# FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /app
WORKDIR /app

# Copy the Mask_RCNN directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
# Compile cocoapi
RUN ./docker_setup.sh

# Specify directories that will be shared with host 
VOLUME /data

# Define environment variable
ENV NAME MaskRCNN

# Expose port for jupyter notebook
# EXPOSE 8888

ENTRYPOINT ["python", "samples/sunrgbd/sunrgbd.py"]
