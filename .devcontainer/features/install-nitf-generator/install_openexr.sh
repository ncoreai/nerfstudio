
#!/bin/bash

# use apt to install the required packages
apt-get update
apt-get install -y cmake libopenexr-dev openexr build-essential 



## Install pynitf python library
# URL of the pynitf source tar.gz file
IMATH_URL="https://github.com/AcademySoftwareFoundation/Imath/archive/refs/tags/v3.1.9.tar.gz"

# The directory where the tar.gz will be downloaded and extracted
DOWNLOAD_DIR="imath_download"

# Create a directory to store the download
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

# Download the pynitf tar.gz file
echo "Downloading pynitf from $IMATH_URL..."
wget $IMATH_URL -O imath.tar.gz

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download pynitf. Please check the URL and your internet connection."
    exit 1
fi

# Extract the tar.gz file but make the directory name Imath_src
echo "Extracting Imath..."
tar -xzvf imath.tar.gz --transform 's,^[^/]\+\($\|/\),Imath_src\1,'


# Navigate into the extracted directory
cd Imath_src


# run cmake 
echo "Running cmake..."
cmake . 
make install

# pip install the openexr python library
pip install git+https://github.com/jamesbowman/openexrpython.git