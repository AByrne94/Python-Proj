import pandas as pd
import numpy as np

print("hello world")

from setuptools import find_packages, setup
setup(
    name='laxleague',
    extras_require=dict(tests=['pytest']),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

