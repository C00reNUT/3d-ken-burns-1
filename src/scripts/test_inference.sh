# !/bin/bash
python ../autozoom.py --in ../images/doublestrike.jpg --out ../movies/doublestrike.mp4

if [ -f "../movies/doublestrike.mp4" ]; then
    echo "Inference Success."
    rm ../movies/doublestrike.mp4
else
    echo "Inference Failure"
fi