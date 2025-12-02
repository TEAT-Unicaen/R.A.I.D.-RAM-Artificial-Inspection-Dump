#!/usr/bin/env python3
"""
Entry point for RAID project.
"""

####Only standard library imports###
import os
import subprocess
import sys
import importlib.util
import torch
####################################

from utils.trainer import trainModel

DEBUG = False

# Tuples list (Name for PIP, Name for IMPORT)
# If import name is the same as package name, use None for the second argument
CONST_DEPENDENCIES = [
    ("python-dotenv", "dotenv"), 
]

def main():
    print("R.A.I.D. Project Entry Point")
    
    # Load environment variables from .env file
    load_dotenv()

    DEBUG = os.getenv("DEBUG") == "True"
    USE_CPU = os.getenv("USE_CPU") == "True"
    if DEBUG:
        print("Debug mode is ON")
        print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)

        print("GPU détecté :", props.name)
        print("VRAM total :", round(props.total_memory / 1e9, 2), "GB")
    else:
        if not USE_CPU:
            raise RuntimeError("GPU not detected. CUDA is not available.")
        else:
            print("No GPU detected. Using CPU as per configuration.")
            device = torch.device("cpu")

    return 0

def verifyAndInstall(package_pip, nom_import=None):
    """
    Verify if a Python module is installed, and install it via pip if not.
    Args:
        package_pip (str): The name of the package to install via pip.
        nom_import (str, optional): The name of the module to import. 
                                    If None, uses package_pip as the import name.
    """
    if nom_import is None:
        nom_import = package_pip

    spec = importlib.util.find_spec(nom_import)
    
    if spec is None:
        print(f"The module '{nom_import}' is missing. Installing '{package_pip}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_pip])
            print(f"{package_pip} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"Error installing {package_pip}.")
    else:
        pass

    trainModel()

if __name__ == "__main__":
    for package, import_name in CONST_DEPENDENCIES:
        verifyAndInstall(package, import_name)

    # All dep import here (after verification) 
    from dotenv import load_dotenv

    raise SystemExit(main())



