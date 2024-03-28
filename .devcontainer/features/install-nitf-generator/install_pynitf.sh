
#!/bin/bash

## Install pynitf python library
# URL of the pynitf source tar.gz file
PYNITF_URL="https://github.com/Cartography-jpl/pynitf/archive/refs/tags/1.13.tar.gz"

# The directory where the tar.gz will be downloaded and extracted
DOWNLOAD_DIR="pynitf_download"

# Create a directory to store the download
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

# Download the pynitf tar.gz file
echo "Downloading pynitf from $PYNITF_URL..."
wget $PYNITF_URL -O pynitf.tar.gz

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download pynitf. Please check the URL and your internet connection."
    exit 1
fi

# Extract the tar.gz file
echo "Extracting pynitf..."
tar -xzvf pynitf.tar.gz

# Navigate into the extracted directory
cd pynitf-1.13

# Install the package using Python
echo "Installing pynitf..."
pip install .

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "pynitf installed successfully."
else
    echo "Failed to install pynitf. Please check for any errors in the output."
fi

# Clean up (optional)
cd ../..
rm -rf $DOWNLOAD_DIR

echo "Installation script completed."
