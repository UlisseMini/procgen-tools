# %% Don't have to restart kernel and reimport each time you modify a dependency
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %% Super-long import code!
from procgen_tools.imports import *
from procgen_tools.procgen_imports import * # TODO doesn't let us autoreload

# %% [markdown]
# Let's see whether the c55 synthetic patch reproduces behavior in the unpatched model. 

# %%
# For each seed, compute the cheese location and then find an appropriate channel patch
