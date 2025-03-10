
#!/bin/bash

# 
cd /workspace/
# Cause the script to exit on failure.
set -eo pipefail

# Activate the main virtual environment
. /venv/main/bin/activate

# Install your packages
pip install polars, pandas, catboost, numpy

# Clone the Git repository (replace with the actual repo URL)
git clone https://github.com/Flash1709/vastai.git
