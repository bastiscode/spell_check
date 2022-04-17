from setuptools import setup

with open("README.rst", "r", encoding="utf8") as rdm:
    long_description = rdm.read()

with open("nsc/version.py", "r", encoding="utf8") as vf:
    version = vf.readlines()[-1].strip().split()[-1].strip("\"'")

setup(
    name="nsc",
    version=version,
    description="Neural spell checking using Transformers and Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Sebastian Walter",
    author_email="swalter@tf.uni-freiburg.de",
    python_requires=">=3.6",
    packages=["nsc"],
    scripts=[
        "bin/nsec",
        "bin/nsed",
        "bin/ntr"
    ],
    install_requires=[
        "torch>=1.11.0",
        "dgl>=0.8.0",
        "einops>=0.3.0",
        "numpy>=1.19.0",
        "tokenizers>=0.10.0",
        "omegaconf>=2.1.1",
        "spacy>=3.2.0",
        "tqdm>=4.49.0",
        "requests>=2.27.0",
        "matplotlib>=3.4.3"
    ],
    extras_require={
        "train": [
            "lmdb>=1.1.0",
            "lz4>=3.1.0"
            "tensorboard>=2.8.0"
        ]
    }
)
