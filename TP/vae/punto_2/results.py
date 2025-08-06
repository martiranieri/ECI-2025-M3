import pandas as pd
import numpy as np

####### métricas #######################################################
df = pd.read_csv("metrics.csv")

for metric in ["nelbo", "kl", "rec"]:
    media = df[metric].mean()
    stderr = df[metric].std(ddof=1) / np.sqrt(len(df))
    print(f"{metric.upper()}: media: {media:.2f}  error estándar:{stderr:.2f}")
