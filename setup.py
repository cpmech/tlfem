# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from setuptools import setup, find_packages

setup(
    name="TlFEM",
    version="1.1.0",
    author="Dorival Pedroso",
    author_email="dorival.pedroso@gmail.com",
    packages=find_packages(),
    package_data={"": ["data/*.msh", "data/*.cmp"]},
    scripts=["scripts/*.py"],
    url="https://github.com/cpmech/tlfem",
    license="LICENSE",
    description="Teaching and Learning the Finite Element Method",
    long_description=open("README.md").read(),
    install_requires=["matplotlib", "numpy", "scipy",],
)
