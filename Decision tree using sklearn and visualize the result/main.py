import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# visualize func
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image  
import pydotplus

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


def split_data(data):
    x = data.iloc[:,1:] # second col to the last
    y = data.iloc[:,0] # "Class" is our target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .28)
    return x_train, x_test, y_train, y_test

def build_decision_tree(x_train, x_test, y_train):
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    return pred, tree

def evaluate(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")

def visualize_tree(tree):
    """StringIO(): creates an object (empty in this case) to receive a string buffer (the tree will be created first as a string before as an image) in DOT (graph description language) format."""
    dot_data = StringIO()
    """export_graphviz(): exports the tree in DOT format, generating a representation of the decision tree, which is written into the ‘out_file’."""
    export_graphviz(tree, out_file = dot_data)
    """graph_from_dot_data(): will use the DOT object to create the graph."""
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(r'C:\Users\USER\Desktop\ML-implementation\Decision tree using sklearn and visualize the result\tree.png')
    Image(graph.create_png())

data = load_data("breast-cancer.data")
x_train, x_test, y_train, y_test = split_data(data)
prediction, tree = build_decision_tree(x_train, x_test, y_train)
evaluate(y_test, prediction)
visualize_tree(tree)




