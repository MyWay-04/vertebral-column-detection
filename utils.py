# utils.py
import os

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

