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
    if DEBUG:
        print("Debug mode is ON")
        print("PyTorch version:", torch.__version__)
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
            print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
            print("CUDA Version:", torch.version.cuda)
        else:
            print("CUDA is not available.")


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

    # Check if the module can be found by Python
    spec = importlib.util.find_spec(nom_import)
    
    if spec is None:
        print(f"The module '{nom_import}' is missing. Installing '{package_pip}'...")
        try:
            # Install the package using pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_pip])
            print(f"{package_pip} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"Error installing {package_pip}.")
    else:
        pass

if __name__ == "__main__":
    #Verify and install dependencies
    for package, import_name in CONST_DEPENDENCIES:
        verifyAndInstall(package, import_name)

    #Import after verification
    from dotenv import load_dotenv

    raise SystemExit(main())



