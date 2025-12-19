import os
import glob
from PIL import Image
import numpy as np

def check_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        0: "saglikli_cilt",
        1: "yanik_1derece",
        2: "yanik_2derece",
        3: "yanik_3decerece"
    }
    
    total = 0
    for label, folder_name in dirs.items():
        folder_path = os.path.join(base_dir, folder_name)
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
            files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        print(f"Sınıf {label} ({folder_name}): {len(files)} dosya")
        total += len(files)
    print(f"Toplam dosya: {total}")

if __name__ == "__main__":
    check_data()
