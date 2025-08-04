import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codebase.train import train  # Si lo necesitás realmente

# --- Parte 1: métricas ---
df = pd.read_csv("metrics.csv")

for metric in ["nelbo", "kl", "rec"]:
    media = df[metric].mean()
    stderr = df[metric].std(ddof=1) / np.sqrt(len(df))
    print(f"{metric.upper()}: media: {media:.2f}  error estándar:{stderr:.2f}")
