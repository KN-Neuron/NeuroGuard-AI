from setuptools import setup, find_packages

with open("neuroguard/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuroguard",
    version="0.0.1",
    author="KN Neuron - AI Team",
    description="A library for EEG-based AI pipelines and models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["neuroguard", "neuroguard.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12, <3.13",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "mne",
        "umap-learn>=0.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "neuroguard=neuroguard.__main__:main",
        ],
    },
    license="MIT",
    keywords="EEG AI machine-learning deep-learning neuroscience",
)
