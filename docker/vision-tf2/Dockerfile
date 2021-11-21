
#Loading a preexisting image, try to get as close as you can before you start building.
FROM tensorflow/tensorflow:2.5.0-gpu

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get install -y \
        git \
        unzip wget

RUN pip install pandas scikit-learn scipy seaborn scikit-image matplotlib

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*
