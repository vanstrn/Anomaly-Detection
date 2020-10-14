define run_command
		docker run -it --rm \
			-u `id -u`:`id -g` \
			--gpus all \
			--memory="10G" --memory-swap="15G" \
			--network=host \
			-v ${PWD}:${PWD} \
			-w ${PWD} \
			raide-rl \
			$(1)
endef

#Can't be the same name as a directory
image:
	docker build --rm -t raide-rl -f docker/raide-rl/Dockerfile docker/raide-rl

dev:
	$(call run_command,bash)

tensorboard:
	docker run -it --rm --name tensorboard \
		--memory="4G" --memory-swap="15G" \
		--network=host \
		-v ${PWD}:${PWD} \
		-w ${PWD} \
		raide-rl \
		tensorboard --logdir=logs

clean:
	rm -r images/ logs/ models/

env:
	conda env create -f rl.yml

# tensorboard:
# 	google-chrome http://localhost:6006
# 	tensorboard --logdir=logs &> /dev/null

cluster:
	ssh nealeav2@cc-login.campuscluster.illinois.edu

.PHONY: install build
