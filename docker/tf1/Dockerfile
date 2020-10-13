
#Loading a preexisting image, try to get as close as you can before you start building.
FROM tensorflow/tensorflow:1.14.0-gpu-py3

#Running an update and upgrade. This is usually required before installing anything new.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        git

#Adding custom software and packages.
RUN pip install gym gym-minigrid matplotlib
RUN pip install gym_cap
RUN pip install pandas scikit-learn scipy

#Adding entrypoint for things that need to be executed when the image is mounted
# into a container. Insert any of those arguments there.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
