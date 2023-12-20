from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import sys

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


root_dir = "datamount/supp_data"
total_files = sum(1 for _, _, files in os.walk(root_dir) for _ in files)

def process_file(file_path):
    with open(file_path, 'r') as f:
        bpp = np.array([float(item) for item in f.read().split()])
        seq_len = int(np.max(bpp))
        bpp = bpp.reshape(-1, 3)
        bpp_matrix = np.zeros((seq_len, seq_len))
        i_indices = bpp[:, 0].astype(int) - 1
        j_indices = bpp[:, 1].astype(int) - 1
        bpp_matrix[i_indices, j_indices] = bpp[:, 2]
        return bpp_matrix

def preprocess_files(root_dir):
    sequence_ids = []
    file_paths = []
    with tqdm(total=2150396, unit="files") as pbar:
        for root, dirs, files in os.walk(root_dir):
            for file in tqdm(files, ascii="=>", disable=True):
                sequence_id = os.path.splitext(file)[0]  # Extract the name without the extension
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    sequence_id = os.path.splitext(file)[0]
                    bpp_matrix = process_file(file_path)
                    npy_path = os.path.join(root, sequence_id + '.npz')  # Save in the same directory as the txt file
                    np.savez_compressed(os.path.join(root, sequence_id + '.npz'), bpp_matrix)
                    sequence_ids.append(sequence_id)
                    file_paths.append(npy_path)
                    pbar.update(1)
    
    pd.DataFrame({'sequence_id': sequence_ids, 'file_path': file_paths}).to_csv('datamount/bpp_index.csv')
           
preprocess_files(root_dir)