"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="distributed_shampoo",
        version="0.1.0",
        description="PyTorch implementation of Distributed Shampoo",
        license="BSD 3-clause",
        packages=find_packages(),
        install_requires=[],
        python_requires=">=3.8",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
