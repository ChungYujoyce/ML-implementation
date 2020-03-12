import pandas as pd
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import numpy as np
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def load_data(file_path):
    f = open(file_path, 'r')
    data = f.read()
    DATA = StringIO(data)
    df = pd.read_csv(DATA, sep=",")
    df.dropna(inplace = True)
    # missing values preprocess (delete in this practice)
    return df

data = load_data("breast-cancer.data")
def split_data(data):
    print(data)

print(split_data(data))
