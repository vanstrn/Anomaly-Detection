define run_command
		docker run -it --rm \
			-u `id -u`:`id -g` \
			--gpus all \
			--network=host \
			-v ${PWD}:${PWD} \
			-v ~/.keras:/.keras \
			-w ${PWD} \
			$(1) \
			$(2)
endef

#Can't be the same name as a directory
image:
	docker build --rm -t vision-tf2 -f docker/vision-tf2/Dockerfile docker/vision-tf2

image-pytorch:
	docker build --rm -t vision-pytorch -f docker/vision-pytorch/Dockerfile docker/vision-pytorch

dev:
	$(call run_command,vision-tf2,bash)

dev-pytorch:
	$(call run_command,vision-pytorch,bash)

tensorboard:
	docker run -it --rm \
		--memory="4G" \
		--network=host \
		-v ${PWD}:${PWD} \
		-w ${PWD} \
		vision-tf2 \
		tensorboard --logdir=logs

clean:
	rm -r logs/ models/

.PHONY: install build
