from typing import List, Tuple, Dict, Union, Optional, Callable
import re
from collections import defaultdict
import pickle
import funcy as fn

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
from ipywidgets import interact
import itertools
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

import circrl.module_hook as cmh
import circrl.rollouts as cro

import procgen_tools.models as models
import procgen_tools.utils as utils
from procgen import ProcgenGym3Env

import os, sys
from glob import glob
from pathlib import Path

from ipywidgets import (
    Text,
)  # Import this later because otherwise Text gets cast as str?

RAND_REGION = 5
NUM_ACTIONS = 15
try:
    get_ipython()
    in_jupyter = True
except NameError:
    in_jupyter = False
# PATH_PREFIX = '../' if in_jupyter else ''


def load_model(rand_region: int = 5, num_actions: int = 15, use_small: bool = False):
    """Load a model from the trained_models folder. Returns the policy and the hook."""
    model_name = "maze_i" if use_small else f"maze_I/model_rand_region_{rand_region}"
    model_stub = f"trained_models/{model_name}.pth"

    try:
        utils.cd_into_procgen_tools()
    except Exception:
        Path("procgen-tools").mkdir(parents=True, exist_ok=True)
        os.chdir("procgen-tools")

    policy = models.load_policy(model_stub, num_actions, t.device("cpu"))
    hook = cmh.ModuleHook(policy)
    return policy, hook


policy, hook = load_model(
    rand_region=RAND_REGION, num_actions=NUM_ACTIONS, use_small=False
)

# Useful general variables
default_layer = "embedder.block2.res1.resadd_out"
hook.run_with_input(np.zeros((1, 3, 64, 64), dtype=np.float32))
labels = list(hook.values_by_label.keys())  # all labels in the model
if "_out" in labels:
    labels.remove("_out")
