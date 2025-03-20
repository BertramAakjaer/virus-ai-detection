@echo off
:: Gather all requirements from requirements.txt
if exist requirements.txt (
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping installation.
)

:: Run the Python script to_csv.py
if exist to_csv.py (
    echo Running to_csv.py...
    python to_csv.py
) else (
    echo to_csv.py not found. Exiting.
)