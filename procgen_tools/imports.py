from typing import List, Tuple, Dict, Union, Optional, Callable
import re 
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import torch as t
import math 

import plotly.express as px
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm
from einops import *
from IPython.display import *
from ipywidgets import *
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 140

from procgen_tools.utils import setup
import procgen_tools.models as models

setup() # create directory structure and download data 

# Install procgen tools if needed
try:
  import procgen_tools
except ImportError:
  get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')
  
import circrl.module_hook as cmh
import circrl.rollouts as cro

from procgen import ProcgenGym3Env

import os, sys
from glob import glob
from pathlib import Path

from ipywidgets import Text # Import this later because otherwise Text gets cast as str?

FOLDER = 'experiments'
if FOLDER not in os.getcwd(): # path this notebook expects to be in
    Path(FOLDER).mkdir(exist_ok=True)
    os.chdir(FOLDER)
RAND_REGION = 5
NUM_ACTIONS = 15
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
PATH_PREFIX = '../' if in_jupyter else ''

# Load model
model_path = PATH_PREFIX + f'trained_models/maze_I/model_rand_region_{RAND_REGION}.pth'
policy = models.load_policy(model_path, NUM_ACTIONS, t.device('cpu'))
hook = cmh.ModuleHook(policy)

# Useful general variables
main_label = 'embedder.block2.res1.resadd_out'
hook.run_with_input(np.zeros((1,3, 64, 64), dtype=np.float32))
labels = list(hook.values_by_label.keys()) # all labels in the model
if '_out' in labels: labels.remove('_out')