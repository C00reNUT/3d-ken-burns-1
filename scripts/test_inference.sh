# !/bin/bash
python ./src/autozoom.py --in ./src/images/doublestrike.jpg --out /src./movies/doublestrike.mp4

if [ -f "./src/movies/doublestrike.mp4" ]; then
    echo "Inference Success."
    rm ./src/movies/doublestrike.mp4
else
    echo "Inference Failure"
fi