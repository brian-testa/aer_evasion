# Run this to start jupyter-lab in the docker container. Useful for debug and to run Jupyter Notebooks
# Note: Readonly for notebooks, but CAN make changes in the data and pickles directories

#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Note: To get to this jupyter-lab instance, copy and paste the URL into your browser and change the port from 8888 to whatever you used in run_batch.sh"
echo "(Default is 10000)"

$SCRIPT_DIR/run_batch.sh jupyter-lab --allow-root
