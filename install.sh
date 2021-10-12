#!/bin/bash

set -e

if [ ! -f /src/selected_models/model.h5 ]
then
    curl -L https://github.com/davidmallasen/LiveChess2FEN/releases/download/v0.1.0/Xception_last.h5 --output /src/selected_models/model.h5
fi

if [ ! -f /src/selected_models/TestImages.zip ]
then
    curl -L https://github.com/davidmallasen/LiveChess2FEN/releases/download/v0.1.0/TestImages.zip --output /src/temp/TestImages.zip

    cd /src/temp && unzip TestImages.zip

    mv /src/temp/TestImages/FullDetection/* /src/predictions/
fi


