from setuptools import setup, find_packages

setup(
    name="transmission-robust-deepfake-detection",
    version="1.0.0",
    author="Aradhya Singh, Khushi Sagar",
    description="Transmission-Robust Deepfake Detection via Latent Feature Regularization and Dynamic Gated Fusion",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AradhyaSingh7/transmission-robust-deepfake-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
    ],
)
