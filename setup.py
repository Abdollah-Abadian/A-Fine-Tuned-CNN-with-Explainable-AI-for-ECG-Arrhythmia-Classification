from setuptools import setup, find_packages

setup(
    name="ftcnn-ecg",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "wfdb>=4.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
)
