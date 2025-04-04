@echo off

:: Run the Python script to_csv.py
if exist change_folder.py (
    echo Running change_folder.py...
    python change_folder.py
) else (
    echo change_folder.py not found. Exiting.
)