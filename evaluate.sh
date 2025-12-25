#!/bin/bash
set -ex

cd /home/someuser

# Verify GPU access first
echo "Verifying GPU access..."
python verify_gpu.py
if [ $? -ne 0 ]; then
    echo "ERROR: GPU verification failed. Please check your GPU setup."
    exit 1
fi

# Run attack and defense in separate processes. This is required since adaptive
# attacks seem to interfere with the gradients in the defense at test time.
python model_server.py &
python evaluate.py

# Note on termination:
# evaluate.py sends a shutdown signal to model_server.py when it's done.
# However, there is no proper error handling so far, so model_server might not
# terminate correctly if there is an error.
# In any case, stopping the Docker container will kill everything.
