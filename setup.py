# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from setuptools import setup, find_packages

setup(
    name="TlFEM",
    version="1.0.0",
    author="Dorival M Pedroso",
    author_email="dorival.pedroso@gmail.com",
    packages=find_packages(),
    package_data={"": ["data/*.msh", "data/*.cmp"]},
    scripts=["scripts/*.py"],
    url="http://code.google.com/p/tlfem/",
    license="LICENSE",
    description="Teaching and Learning the Finite Element Method.",
    long_description=open("README.md").read(),
    install_requires=["matplotlib >= 1.4.2", "numpy >= 1.8.2", "scipy >= 0.14.1",],
)
