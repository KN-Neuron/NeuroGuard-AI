from setuptools import setup, find_packages

with open("eeg_lib/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eeg_lib",
    version="0.0.1",
    author="KN Neuron - AI Team",
    # author_email="your_email@example.com",
    description="A library for EEG-based AI pipelines and models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/your_username/eeg_lib",
    packages=find_packages(include=["eeg_lib", "eeg_lib.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
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
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "eeg_lib=eeg_lib.__main__:main",
        ],
    },
    license="MIT",
    keywords="EEG AI machine-learning deep-learning neuroscience",
)
