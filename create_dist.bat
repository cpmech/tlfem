@echo off
del MANIFEST
c:\Python27\python.exe setup.py bdist_wininst
pause
