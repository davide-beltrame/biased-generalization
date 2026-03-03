from setuptools import setup, find_packages

setup(
    name='biased-generalization',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.1',
        'torchvision',
        'numpy',
        'scipy',
        'numba',
        'matplotlib',
        'seaborn',
        'wandb',
        'tqdm',
    ],
    author='Luca Biggio, Davide Beltrame',
    description=(
        'Code for "Biased Generalization in Diffusion Models" — '
        'diffusion modules, belief propagation, and experiment scripts.'
    ),
    include_package_data=True,
)