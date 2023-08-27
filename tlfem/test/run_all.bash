#!/bin/bash

set -e

python3 t_beam.py
python3 t_defgrad.py
python3 t_mesh.py
python3 t_patchrecov.py
python3 t_porous.py
python3 t_solver.py

