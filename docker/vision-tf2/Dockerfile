
#Loading a preexisting image, try to get as close as you can before you start building.
FROM tensorflow/tensorflow:2.9.1-gpu

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get install -y \
        git \
        unzip wget

RUN pip install pandas scikit-learn scipy seaborn scikit-image matplotlib gitpython
RUN pip install moviepy
