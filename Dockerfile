FROM tensorflow/tensorflow:2.5.0-gpu

RUN mkdir /src
WORKDIR /src

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN apt-get install -y python3-pip
RUN pip3 install -U pip testresources setuptools==49.6.0
RUN pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==3.1.0 keras_preprocessing==1.1. keras_applications==1.0.8 gast==0.4.0 futures protobuf pybind11
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow

RUN apt-get install -y build-essential libatlas-base-dev gfortran
RUN pip3 install keras

RUN pip3 install onnxruntime sklearn numpy scipy joblib matplotlib scikit-image pandas pillow tqdm pyclipper opencv-python tensorflow-gpu nvidia-pyindex

RUN pip3 install tensorboard==2.6.0 torch torchvision torchaudio

RUN apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

RUN mkdir /opencv_build
RUN cd /opencv_build && git clone https://github.com/opencv/opencv.git
RUN cd /opencv_build && git clone https://github.com/opencv/opencv_contrib.git
RUN cd /opencv_build/opencv && git checkout 4.5.0
RUN cd /opencv_build/opencv_contrib && git checkout 4.5.0
RUN mkdir /opencv_build/opencv/build
RUN cd /opencv_build/opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    # -D WITH_CUDA= \
    # -D CUDA_ARCH_BIN="5.3,6.2,7.2" \
    # -D CUDA_ARCH_PTX="" \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_build/opencv_contrib/modules .. && \
    make -j8 && \
    make install

RUN apt-get install -y curl