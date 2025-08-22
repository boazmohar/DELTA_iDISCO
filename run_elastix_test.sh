#!/bin/bash

# Directory where elastix binaries are (or should be) located
ELASTIX_DIR="/groups/spruston/home/moharb/elastix/bin"

# Create directory if it doesn't exist
if [ ! -d "$ELASTIX_DIR" ]; then
    echo "Creating directory: $ELASTIX_DIR"
    mkdir -p "$ELASTIX_DIR"
fi

# Check if elastix is installed
if [ ! -f "$ELASTIX_DIR/elastix" ]; then
    echo "Elastix not found. Would you like to download and install it? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Downloading and installing elastix..."
        
        # Create a temporary directory for download
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR" || exit 1
        
        # Download elastix (adjust URL for your system)
        wget -q --show-progress https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.gz
        
        # Extract it
        tar -xzf elastix-5.0.1-linux.tar.gz
        
        # Create lib directory if it doesn't exist
        ELASTIX_LIB_DIR="${ELASTIX_DIR%/bin}/lib"
        mkdir -p "$ELASTIX_LIB_DIR"
        
        # Copy binaries to the destination
        cp -v bin/elastix "$ELASTIX_DIR/"
        cp -v bin/transformix "$ELASTIX_DIR/"
        
        # Copy libraries as well
        cp -v lib/* "$ELASTIX_LIB_DIR/"
        
        # Clean up
        cd - || exit 1
        rm -rf "$TMP_DIR"
        
        echo "Elastix installed to $ELASTIX_DIR"
    else
        echo "Elastix installation skipped. The test will likely fail."
    fi
fi

# Set the environment variables
export ELASTIX_PATH="$ELASTIX_DIR"
echo "Setting ELASTIX_PATH=$ELASTIX_PATH"

# Also set LD_LIBRARY_PATH to include the lib directory
ELASTIX_LIB_DIR="${ELASTIX_DIR%/bin}/lib"
if [ ! -d "$ELASTIX_LIB_DIR" ]; then
    echo "Creating lib directory: $ELASTIX_LIB_DIR"
    mkdir -p "$ELASTIX_LIB_DIR"
fi
export LD_LIBRARY_PATH="$ELASTIX_LIB_DIR:$LD_LIBRARY_PATH"
echo "Setting LD_LIBRARY_PATH to include: $ELASTIX_LIB_DIR"

# Use the specific h5 file
H5_FILE="/nearline/spruston/Boaz/DELTA/I2/20250111_IDISCO_MouseCity3/ANM555974/uni_tp-0_ch-0_st-0-x00-y00_obj-right_cam-long_etc.lux.h5"

if [ ! -f "$H5_FILE" ]; then
    echo "Error: The specified h5 file does not exist: $H5_FILE"
    exit 1
fi

echo "Using h5 file: $H5_FILE"

# Run the test
echo "Running elastix test..."
python test_elastix.py --elastix_path "$ELASTIX_PATH" --test_h5_file "$H5_FILE"

# Show the exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with exit code $STATUS"
fi