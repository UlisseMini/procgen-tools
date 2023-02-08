# Run with docker run -it -p 8888:8888 valvate/procgen-tools:latest
FROM pytorch/pytorch:latest

WORKDIR /root

RUN apt-get update -y
RUN apt-get install -y git wget vim tmux

# RUN git clone --depth 1 https://github.com/JacobPfau/procgenAISC
# RUN git clone --depth 1 https://github.com/jbkjr/train-procgen-pytorch
RUN git clone https://github.com/UlisseMini/procgen-tools

WORKDIR /root/procgen-tools

RUN pip install -e .
RUN pip install -r requirements.txt

# Download data & setup files

RUN mkdir -p data/vfields figures trained_models/maze_I
RUN wget https://nerdsniper.net/mats/episode_data.tgz && tar -xvzf episode_data.tgz
RUN wget https://nerdsniper.net/mats/data.tgz && tar -xvzf data.tgz
RUN wget https://nerdsniper.net/mats/model_rand_region_5.pth -O trained_models/maze_I/model_rand_region_5.pth

# Jupyter entrypoint

ENTRYPOINT ["jupyter-notebook", ".", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]
