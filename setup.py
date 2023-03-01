from setuptools import setup, find_packages

setup(
    name='procgen-tools',
    description='Tools for working with Procgen environments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        "gym==0.26.*",
        "gym3==0.3.3",
        'numpy>=1.22.1',
        'torch>=1.13.1',
        "matplotlib==3.3.2", # colab broken on latest :(
        "tqdm==4.64.1",
        "ipywidgets==7.*", # v8 causes vscode bug
        "procgen==0.10.7",
        "bidict>=0.22.0",
        "requests==2.28.*",
        "pandas>=1.5.3",
        "circrl>=0.4.0",
        "plotly>=5.13.0",
        "networkx>=3.0",
        "scipy==1.10.0",
        "mistune==0.8.4", # intermediate dependency issue with ndconvert?
        "captum>=0.6.0",
        "kaleido>=0.2.1",
        "funcy>=1.18",
    ]
)
