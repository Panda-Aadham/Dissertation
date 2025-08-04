# To build the image:
docker build -f Dockerfile.tflm-esp32s3 -t tflm-esp32 .

# To run the container
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  tflm-esp32



# To setup test
cd /workspace/esp32-tflm-test

# Set target
idf.py set-target esp32s3

# Configure project
idf.py menuconfig  # Optional: adjust settings

# Build
idf.py build

# Flash (if you have hardware connected)
idf.py -p /dev/ttyUSB0 flash monitor