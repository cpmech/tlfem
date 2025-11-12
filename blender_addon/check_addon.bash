#!/bin/bash

/opt/blender/blender --background --python-expr 'import sys,traceback; sys.path.insert(0, "/home/dorival/01-Code/python");
try:
    import tlfem.blender_addon as mod
    print("MODULE_IMPORT_OK")
    mod.register(); print("MODULE_REGISTER_OK")
except Exception:
    traceback.print_exc()'