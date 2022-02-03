# Sample script showing how to execute batch calls in the eractive session in the docker container
# Typical runtime script for batch processing; you wil need to change paths to match your local machine and may need to change the port below

#!/bin/bash

# CHANGE ME
HOME=/home/brian/Workspace/clean_audio/aer_evasion

docker run -p 10000:8888 -v $HOME/pickles:/pickles -v $HOME/data:/data -v $HOME/scripts:/scripts -v $HOME/notebooks:/notebooks audio $@
