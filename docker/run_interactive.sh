# Sample script showing how to start an interactive session in the docker container
# Good for debugging; you wil need to change paths to match your local machine and may need to change the port below

#!/bin/bash

# CHANGE ME
HOME=/home/brian/Workspace/clean_audio/aer_evasion

docker run -p 10000:8888 -it -v $HOME/pickles:/pickles -v $HOME/data:/data -v $HOME/scripts:/scripts -v $HOME/notebooks:/notebooks audio

