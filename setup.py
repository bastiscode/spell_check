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
        "torch==1.11.0",
        "dgl-cu113",
        "einops==0.4.1",
        "numpy==1.21.2",
        "tokenizers==0.12.1",
        "omegaconf==2.1.2",
        "spacy==3.2.4",
        "tqdm==4.62.3",
        "requests==2.27.1"
    ],
    extras_require={
        "train": [
            "lmdb==1.3.0",
            "lz4==4.0.0",
            "tensorboard==2.8.0"
        ],
        "index": [
            "nmslib==2.1.1",
            "marisa-trie==0.7.7"
        ]
    }
)
