
#!/bin/bash
set -ve
set -x
# set -o pipefail
## Install nitf_generator python library
# URL of the nitf_generator source tar.gz file

NITF_GEN_URL="https://github.com/ncoreai/nitf-generator/archive/refs/heads/main.tar.gz"

# The directory where the tar.gz will be downloaded and extracted
DOWNLOAD_DIR="nitf_generator_download"

# Create a directory to store the download
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

# Download the nitf_generator tar.gz file
echo "Downloading nitf_generator from $nitf_generator_URL..."
# echo $GITHUB_TOKEN
echo "Authorization: token $GITHUB_TOKEN" 
wget --header="Authorization: token $GITHUB_TOKEN" $NITF_GEN_URL -O nitf_generator.tar.gz


# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download nitf_generator. Please check the URL and your internet connection."
    exit 1
fi

# Extract the tar.gz file
echo "Extracting nitf_generator..."
tar -xzvf nitf_generator.tar.gz

# Navigate into the extracted directory
cd nitf-generator-main
# Install the package using Python
echo "Installing nitf_generator..."
pip install .

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "nitf_generator installed successfully."
else
    echo "Failed to install nitf_generator. Please check for any errors in the output."
fi

# Clean up (optional)
cd ../..
rm -rf $DOWNLOAD_DIR

echo "Installation script completed."
