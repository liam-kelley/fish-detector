import os
from util.path_util import deep_glob
from typing import List

def clean_from_suffixes(suffixes: List[str], path: str = "./"):
    # Get all files
    files=[]
    for suffix in suffixes:
        files.extend(deep_glob(path,suffix))
    
    # Remove all files
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    print(f"Removed all files with {suffixes} suffixes.")
    
if __name__ == "__main__":
    clean_from_suffixes(suffixes=["npy", "Identifier"])