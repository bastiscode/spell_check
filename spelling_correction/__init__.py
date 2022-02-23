import os

BASE_DIR = os.path.dirname(__file__)

BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmarks")

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
MISSPELLINGS_DIR = os.path.join(DATA_DIR, "misspellings")
DICTIONARIES_DIR = os.path.join(DATA_DIR, "dictionaries")
