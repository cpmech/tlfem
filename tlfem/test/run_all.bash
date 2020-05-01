#!/bin/bash

set -e

python t_beam.py
python t_defgrad.py
python t_mesh.py
python t_patchrecov.py
python t_porous.py
python t_solver.py
