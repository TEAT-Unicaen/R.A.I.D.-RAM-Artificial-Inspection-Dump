docker build -t raid-project-5080 .

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v "%cd%:/workspace" raid-project-5080 /bin/bash