import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from pandas import DataFrame
import numpy as np
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

# load data and preprocess
def load_data(file_path):
    f = open(file_path, 'r')
    data = f.read()
    DATA = StringIO(data)
    df = pd.read_csv(DATA, sep=",")
    # missing values preprocess (delete in this practice)
    df.dropna(inplace = True)
    le = LabelEncoder()
    df = pd.DataFrame(df)
    df.columns = ["Class", "age", "menopause", "tumor-size" ,"inv-nodes", "node-caps",
   "deg-malig", "breast", "breast-quad", "irradiat"]
    df['breast'] = le.fit_transform(df['breast']) 
    df['menopause'] = le.fit_transform(df['menopause']) 
    df['node-caps'] = le.fit_transform(df['node-caps']) 
    df['breast-quad'] = le.fit_transform(df['breast-quad']) 
    df['age'] = le.fit_transform(df['age']) 
    df['inv-nodes'] = le.fit_transform(df['inv-nodes']) 
    df['tumor-size'] = le.fit_transform(df['tumor-size']) 
    df['irradiat'] = le.fit_transform(df['irradiat']) 
    df['Class'] = le.fit_transform(df['Class']) 
    return df

data = load_data("breast-cancer.data")
def split_data(data):
    print(data)

print(split_data(data))
