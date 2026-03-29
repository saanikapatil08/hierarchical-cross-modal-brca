"""
setup.py
========
Package setup for HCMT project.

Install in editable mode for development:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name='hcmt',
    version='1.0.0',
    description='Hierarchical Cross-Modal Transformer for Breast Cancer Subtype Classification',
    author='[Your Name]',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'einops>=0.7.0',
        'omegaconf>=2.3.0',
        'tqdm>=4.66.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
        ],
        'imaging': [
            'openslide-python>=1.3.0',
            'SimpleITK>=2.3.0',
            'pydicom>=2.4.0',
        ],
        'tracking': [
            'wandb>=0.16.0',
            'tensorboard>=2.14.0',
        ],
    },
)
