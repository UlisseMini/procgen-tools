from setuptools import setup, find_packages

setup(
    name='procgen-tools',
    description='Tools for working with Procgen environments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "gym==0.21.0",
        "gym3==0.3.3",
        "torch==1.13.1",
        "numpy==1.24.1",
        "matplotlib==3.6.3",
        "tqdm==4.64.1",
        "ipywidgets==7.*", # v8 causes vscode bug
        "procgen==0.10.7"
    ]
)
