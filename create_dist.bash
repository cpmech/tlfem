#!/bin/bash

rm -rf build dist TlFEM.egg-info
rm -f MANIFEST
python setup.py sdist
