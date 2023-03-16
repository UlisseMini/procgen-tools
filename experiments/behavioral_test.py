# %% [markdown]
# This notebook is for taking statistics over thousands of runs, in order to analyze which maze features (e.g. distance to cheese) tend to affect decision-making. 

# %%
%load_ext autoreload
%autoreload 2
# %%
try:
    import procgen_tools
except ImportError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

# %%
%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
from procgen import ProcgenGym3Env
from procgen_tools import maze
from procgen_tools.models import load_policy
from procgen_tools.metrics import metrics, decision_square 
from procgen_tools.data_utils import load_episode
from data_util import load_episode

from IPython import display
from glob import glob
import pickle
from tqdm import tqdm

import os
from collections import defaultdict

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split

import random 
from typing import List, Tuple, Any, Dict, Union, Optional

import prettytable 

# %%
os.getcwd()

# %%
files = glob('experiments/statistics/data/*rand_region_5*.pkl') # 80% this won't change qualitative conclusions from earlier runs 
# TODO this requires ./data_util.py to be in main directory; rerun stats to fix this problem? 
runs = []
for f in files:
    try:
        runs.append(load_episode(f, load_venv=False))
    except (AssertionError, KeyError) as e:
        print(f"Malformed file {f}: {e}")
        #os.remove(f)

print(f'Loaded {len(runs)} runs')

# %%
recorded_metrics = defaultdict(list)
recorded_runs = []
got_cheese = []
for run in tqdm(runs):
    g = run.grid()
    if decision_square(g) is None or (g[-5:, -5:] == maze.CHEESE).any():
        continue
    for name, metric in metrics.items():
        recorded_metrics[name].append(metric(g))
    got_cheese.append(float(run.got_cheese))
    recorded_runs.append(run)

runs = recorded_runs; del recorded_runs
got_cheese = np.array(got_cheese)
len(got_cheese)

# TODO only examine the filtered runs 

# %%
# We want to turn the metrics into a dataframe, so we have to convert them to numpy arrays
for name, metric in recorded_metrics.items():
    recorded_metrics[name] = np.array(metric)

# %%
prob = sum(got_cheese) / len(got_cheese)
print(f'P(get cheese | decision square, cheese not in top 5x5) = {prob:.3f}')

# %%
# We filter the data based on the conditions. In particular, we are filtering the data based on the distance between the cheese and the decision square, and the number of steps between the cheese and the decision square. We are also filtering the data based on the fact that the cheese is not in the top 5x5 region, and for mazes which have a decision square
cond = np.logical_and(
    (recorded_metrics['euc_dist_cheese_decision_square'] > 0),
    (recorded_metrics['steps_between_cheese_decision_square'] > 20),
) 

basenum= len (np.nonzero(cond)[0])

sp_indexes = np.nonzero(cond)[0]
filtered_rm = {}
for key, value in recorded_metrics.items():
    filtered_rm[key] = recorded_metrics[key][sp_indexes]

assert len(filtered_rm['euc_dist_cheese_decision_square']) == len(sp_indexes)

# %%
# Make a plotly histogram, where you select which metric to display using a dropdown menu
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import math

fig = go.FigureWidget()

@interact 
def show_histogram(metric=list(recorded_metrics.keys())):
    """ Show a histogram of the metric on this dataset. """
    fig.data = []

    # Add a trace to fig
    fig.add_trace(go.Histogram(x=recorded_metrics[metric], histnorm='probability density'))

    # Set the title
    fig.update_layout(title_text=f'Histogram of {metric}')
    # Set y axis label to probability density
    fig.update_yaxes(title_text='Probability density') 
    fig.update_xaxes(title_text=metric)

    # Automatically display updates to fig without having to call fig.show()
    display.display(fig)

# %%
from ipywidgets import Dropdown, Checkbox

scatter_distances_fig = go.FigureWidget()

# Make plotly scatterplot comparing two metrics, to check for collinearity
@interact
def show_scatter(metric1=Dropdown(options=list(filtered_rm.keys()), value='euc_dist_cheese_decision_square'), 
                 metric2=Dropdown(options=list(filtered_rm.keys()), value='steps_between_cheese_decision_square'),
                 filter_cheese=Checkbox(value=True, description='Filter trivial cases')):
    """ Show a scatterplot of two metrics on this dataset. """
    # Choose which data to use
    data = filtered_rm if filter_cheese else recorded_metrics

    # Plot the scatterplot
    scatter_distances_fig.data = []
    scatter_distances_fig.add_trace(go.Scatter(x=data[metric1], y=data[metric2], mode='markers', name='runs'))
    scatter_distances_fig.update_layout(title_text=f'{metric1} vs {metric2}')
    scatter_distances_fig.update_xaxes(title_text=metric1)
    scatter_distances_fig.update_yaxes(title_text=metric2)
    display.display(scatter_distances_fig)

    # Draw a line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[metric1], data[metric2])
    scatter_distances_fig.add_trace(go.Scatter(x=data[metric1], y=slope*data[metric1] + intercept, mode='lines', name='best fit'))

    # Hide the legend
    scatter_distances_fig.update_layout(showlegend=False)

    # Print the correlation coefficient
    print(f"Correlation between {metric1} and {metric2}: {np.corrcoef(data[metric1], data[metric2])[0,1]:.3f}")

# %%
# Show the correlation matrix in plotly
final_metrics = ['euc_dist_cheese_decision_square']

correlations = np.zeros((len(filtered_rm), len(filtered_rm)))
for i, metric1 in enumerate(filtered_rm.keys()):
    for j, metric2 in enumerate(filtered_rm.keys()):
        correlations[i, j] = np.corrcoef(filtered_rm[metric1], filtered_rm[metric2])[0,1]

# Show the correlation matrix in plotly, with a colorbar 
# On mouse over, show the name of each metric
corrmap = px.imshow(correlations, labels=dict(x='Metric 1', y='Metric 2', color='Correlation'), color_continuous_scale='RdBu', zmin=-1, zmax=1)
# Hover template: show the name of each metric, by looking up its value in the x and y lists 
corrmap.update_traces(hovertemplate='(%{x}, %{y}) = %{z:.3f} <extra></extra>')

corrmap.update_layout(title_text='Correlation matrix between metrics')
# Don't show numbers over each cell
corrmap.update_traces(text=None)
# Show x and y axis labels
corrmap.update_xaxes(ticktext=list(filtered_rm.keys()), tickvals=list(range(len(filtered_rm.keys()))))
corrmap.update_yaxes(ticktext=list(filtered_rm.keys()), tickvals=list(range(len(filtered_rm.keys()))))

# Hide x and y axis titles
corrmap.update_xaxes(title_text='')
corrmap.update_yaxes(title_text='')  

# Hide the color bar 
corrmap.update_layout(coloraxis_showscale=False)

# Make size of plot a bit bigger
corrmap.update_layout(width=800, height=800)
corrmap.show()

# %%
# Get the top k absolute value correlations, ignoring diagonals
k = 50
len_diagonal = len(correlations)
topk = np.argsort(np.abs(correlations).flatten())[-(k+len_diagonal):][::-1] # Diagonal will be 1, so just add in len_diagonal

# Print the top k correlations in a pretty table
table = prettytable.PrettyTable() 
table.field_names = ["Metric 1", "Metric 2", "Correlation"]
for i in topk:
    # Get the row and column of the correlation
    row, col = i // len(correlations), i % len(correlations)
    # Ignore the diagonal and the lower triangle
    if row >= col:
        continue
    # Get the metric names
    metric1, metric2 = list(filtered_rm.keys())[row], list(filtered_rm.keys())[col]
    # Add the row to the table
    table.add_row([metric1, metric2, correlations[row, col]])

# Print the table TODO show floats to 3 decimal places
print(table)

# %%
def run_regression(attrs : List[str], data_frame : pd.DataFrame):
    """ Runs a LASSO-regularized regression on the data using the given attributes. Returns the clf. """ 
    assert len(attrs) > 0, "Must have at least one attribute to regress upon"
    # Ensure attrs is in data_frame
    for attr in attrs:
        assert attr in data_frame, f"Attribute {attr} not in data frame"
    assert 'cheese' in data_frame, "Attribute 'cheese' not in data frame"
    
    x = data_frame[attributes] 
    y = np.ravel(data_frame[['cheese']])

    clf = LogisticRegression(random_state=0, solver ='liblinear', penalty= 'l1').fit(x, y)
    return clf

def display_coeff_table(clf : Any, attrs : List[str]):
    """ Displays the coefficients for each attribute, printing the label next to each coefficient. """ 
    assert len(attrs) > 0, "Must have at least one attribute"

    # Print the coefficient for each attribute, printing the label next to each coefficient
    table = prettytable.PrettyTable()
    table.field_names = ["Attribute", "Coefficient"]
    for i, attr in enumerate(attrs):
        table.add_row([attr, clf.coef_[0][i]])

    # Add a row for the intercept
    table.add_row(["Intercept", clf.intercept_[0]])
    print(table)

# %%
keys = list(filtered_rm.keys())

# Data will track the data for each run, and filtered_data will track the data for each run that got cheese
data = { key: recorded_metrics[key] for key in keys }
filtered_data = { key: filtered_rm[key] for key in keys }

df = pd.DataFrame(data) 
filtered_df= pd.DataFrame(filtered_data) 

df= stats.zscore(df) # zscore standardizes the data by subtracting the mean and dividing by the standard deviation
filtered_df= stats.zscore(filtered_df)

# Now we want to add the cheese column to the dataframe
df ['cheese'] = pd.DataFrame({'cheese': [(runs[i].got_cheese) for i in range(len(runs))]})
filtered_df ['cheese'] = pd.DataFrame({'cheese': [(runs[i].got_cheese) for i in sp_indexes]})

# Choose which keys to regress upon
attributes = [
    'steps_between_cheese_5x5', 
    'euc_dist_cheese_5x5',
    'steps_between_decision_square_5x5',
    'euc_dist_decision_square_5x5',
    'steps_between_cheese_top_right',
    'euc_dist_cheese_top_right',
    'steps_between_decision_square_top_right',
    'euc_dist_decision_square_top_right',
    'steps_between_cheese_decision_square',
    'euc_dist_cheese_decision_square',
    # 'cheese_coord'
    ] 

claimed_attributes = ['steps_between_cheese_decision_square', 'euc_dist_cheese_decision_square','euc_dist_cheese_top_right', 'euc_dist_decision_square_5x5']

n_runs = 50
total_score = 0

# We reduce variance in the score by running the regression multiple times
for x in range(n_runs): 
    train, test = train_test_split(filtered_df, test_size=0.2)

    clf = run_regression(attributes, train)

    x = test[attributes]
    y = np.ravel(test[['cheese']])

    total_score += clf.score(x, y)

# Print the coefficient for each attribute, printing the label next to each coefficient (for the last run)
display_coeff_table(clf, attributes)
print("The average score is ", total_score/n_runs) # NOTE is this avg accuracy?

regression_coeff_signs = { key: (clf.coef_[0][i] > 0) for i, key in enumerate(claimed_attributes) }

# %%
# Let's see how robust the signs are to regressing upon a random subset of attributes
def run_subset_regression(data_frame : pd.DataFrame, n_attrs : int, show_table : bool = False) -> Tuple[Any, List[str]]:
    """ Runs a regression on the data frame using a random subset of n_attrs attributes. Returns the clf and the attributes used. """
    attrs = random.sample(list(filtered_rm.keys())[:-1], n_attrs)
    attrs += claimed_attributes
    clf = run_regression(attrs, train)
    if show_table: display_coeff_table(clf, attrs)

    return clf, attrs

def check_claimed_signs(clf : Any, attrs : List[str], data_frame : pd.DataFrame, show_table : bool = False) -> Dict[str, int]:
    """ Checks if the signs of the regression coefficients for the given attributes match regression_coeff_signs. Returns a dictionary which counts the number of times each attribute had the wrong sign. """
    counters = defaultdict(int)
    for i, attr in enumerate(attrs):
        if attr not in regression_coeff_signs.keys(): continue
        assert attr in data_frame, f"Attribute {attr} not in data frame"
        if (clf.coef_[0][i] >= 0) != regression_coeff_signs[attr]:
            print(f"Attribute {attr} has incorrect sign; expected {regression_coeff_signs[attr]} but got {clf.coef_[0][i] >= 0}")
            if show_table: display_coeff_table(clf, attrs)
            counters[attr] += 1
            # Return the attribute that had the incorrect sign
    return counters

# Run the regression multiple times and check the signs
# See distribution of sign errors over multiple runs
counter = defaultdict(int)
n_rand_runs = 100 # TODO check what this means
for x in range(n_rand_runs):
    clf, attrs = run_subset_regression(train, 4)
    new_counter = check_claimed_signs(clf, attrs, train, show_table=True)

    for key in new_counter.keys():
        counter[key] += new_counter[key] / n_rand_runs
print(counter)

# %%
# Let's compute VIF (variance inflation factor) for each attribute
# This is a measure of how much the variance of the coefficient is inflated due to multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ignore runtime warnings due to division by 0 from R2=1
import warnings
warnings.filterwarnings("ignore")

for attrs in [claimed_attributes, filtered_rm.keys()]:
    print()
    print("VIF for some attributes" if attrs == attributes else "VIF for all attributes")
    vif = pd.DataFrame()
    vif["Features"] = df[attrs].columns
    vif["VIF"] = [variance_inflation_factor(df[attrs].values, i) for i in range(df[attrs].shape[1])]

    print(vif)



# %%
