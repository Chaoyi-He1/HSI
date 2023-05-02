from PIL import Image
import torch
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import random
from tqdm import tqdm


data_path = "./SVR_weights/model_41_linear_weight.csv"
data = np.array(pd.read_csv(data_path, header=None))

# weight = np.sum(np.abs(data), axis=0)
weight = np.sum(data, axis=0)
index = np.flip(np.argsort(weight))
print(index)
print(weight[index])
