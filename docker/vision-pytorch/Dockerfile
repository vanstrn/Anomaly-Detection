
#Loading a preexisting image, try to get as close as you can before you start building.
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get install -y \
        git \
        unzip wget

RUN pip install pandas==1.3.2 scikit-learn scipy seaborn matplotlib gitpython \
    tqdm scikit-image torchsummary
