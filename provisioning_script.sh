
#!/bin/bash

# 
cd /workspace/
# Cause the script to exit on failure.
set -eo pipefail

# Activate the main virtual environment
. /venv/main/bin/activate

# Install your packages
pip install polars pandas catboost numpy gputil gdown pyarrow

# Clone the Git repository (replace with the actual repo URL)
git clone https://github.com/Flash1709/vastai.git

cd /vastai

# download dataset
gdown "https://drive.google.com/uc?id=1r9aWhgK_3yMetmj3I51O1E7krTgCeKx1" --fuzzy
