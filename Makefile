define run_command
		docker run -it --rm \
			-u `id -u`:`id -g` \
			--gpus all \
			--memory="10G" --memory-swap="15G" \
			--network=host \
			-v ${PWD}:${PWD} \
			-w ${PWD} \
			vision-tf2 \
			$(1)
endef

#Can't be the same name as a directory
image:
	docker build --rm -t vision-tf2 -f docker/vision-tf2/Dockerfile docker/vision-tf2

dev:
	$(call run_command,bash)

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
