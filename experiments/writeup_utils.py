import os
from subprocess import run


def r(cmd: str):
    if os.environ.get('TERM').startswith('xterm'):
        print(f'\x1b[31m$ {cmd}\x1b[0m', flush=True, file=os.sys.stderr)
    else:
        print(f'$ {cmd}', flush=True, file=os.sys.stderr)

    return run(cmd, shell=True, check=True)


def setup_colab():
    r('git clone https://github.com/UlisseMini/procgen-tools')
    os.chdir('procgen-tools')
    r('pip install -e .')
    r('pip install -r requirements.txt')
    r('mkdir -p data/vfields figures trained_models/maze_I')
    r('wget https://nerdsniper.net/mats/episode_data.tgz && tar -xvzf episode_data.tgz')
    r('wget https://nerdsniper.net/mats/data.tgz && tar -xvzf data.tgz')
    r('wget https://nerdsniper.net/mats/model_rand_region_5.pth -O trained_models/maze_I/model_rand_region_5.pth')
    r('tar xf data.tgz')
    os.chdir('experiments')

