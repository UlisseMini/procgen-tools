from glob import glob
from tqdm import tqdm
import pickle
from data_util import Episode

for fname in tqdm(glob('../episode_data/*.pkl')):
    try:
        # episode = load_episode(fname)
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, Episode):
            with open(fname, 'wb') as f:
                pickle.dump(obj.__getstate__(), f)
    except ImportError:
        print(f'ImportError {fname}: Does data_util.py exist?')
        exit()
