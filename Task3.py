
#Importing data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
data=pd.read_csv("cardio_train.csv",delimiter=';').drop(['id'],axis=1)

#Preprocessing Data
m_age=min(data['age'])
t_age=max(data['age'])
amount=int((t_age+m_age)/5)

binInterval = [m_age, m_age+amount,(m_age+(2*amount)), (m_age+(3*amount)), (m_age+(4*amount)),(m_age+(5*amount))]
binLabels   = [1, 2, 3, 4,5]
data['age'] = pd.cut(data['age'], bins = binInterval, labels=binLabels)

m_h=min(data['height'])
t_h=max(data['height'])
amount_h=int((t_h+m_h)/4)
binInterval_h = [m_h, m_h+amount_h,(m_h+(2*amount_h)), (m_h+(3*amount_h)),(m_h+(4*amount_h))]
binLabels_h= [1, 2, 3,4]
data['height'] = pd.cut(data['height'], bins = binInterval_h, labels=binLabels_h)

m_w=min(data['weight'])
t_w=max(data['weight'])
amount_w=int((t_w+m_w)/4)
binInterval_w = [m_w, m_w+amount_w,(m_w+(2*amount_w)), (m_w+(3*amount_w)),(m_w+(4*amount_w))]
binLabels_w= [1, 2, 3,4]
data['weight'] = pd.cut(data['weight'], bins = binInterval_w, labels=binLabels_w)

data['ap_lo']=abs(data['ap_lo'])
data['ap_hi']=abs(data['ap_hi'])

for row in data['ap_hi']:
    if (row >200):
       
        s=str(row)
        data['ap_hi']=data['ap_hi'].replace(to_replace=row,value=int(s[0:3]))
        if (int(s[0:3])>200):
            data['ap_hi']=data['ap_hi'].replace(to_replace=int(s[0:3]),value=int(s[0:2]))
for row in data['ap_lo']:
    if (row >200):
        s=str(row)
        data['ap_lo']=data['ap_lo'].replace(to_replace=row,value=int(s[0:3]))
        if (int(s[0:3])>200):
            data['ap_lo']=data['ap_lo'].replace(to_replace=int(s[0:3]),value=int(s[0:2]))
    
m_al=min(data['ap_lo'])
t_al=max(data['ap_lo'])
m_ah=min(data['ap_hi'])
t_ah=max(data['ap_hi'])
amount_ah=int((t_ah+m_ah)/3)
binInterval_ah = [m_ah, m_ah+amount_ah,(m_ah+(2*amount_ah)),(m_ah+(3*amount_ah))]
binLabels_ah= [1, 2,3]
data['ap_hi'] = pd.cut(data['ap_hi'], bins = binInterval_ah, labels=binLabels_ah)


amount_al=int((t_al+m_al)/3)
binInterval_al = [m_al, m_al+amount_al,(m_al+(2*amount_al)),(m_al+(3*amount_al))]
binLabels_al= [1, 2,3]
data['ap_lo'] = pd.cut(data['ap_lo'], bins = binInterval_al, labels=binLabels_al)
data=data.dropna()
#Test and Train Split
X = data.iloc[:, :11]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, shuffle=True, random_state=24)

def gini_impurity(label, label_idx):
   
    # the unique labels and counts in the data
    unique_label, unique_label_count = np.unique(label.loc[label_idx], return_counts=True)
    impurity = 1.0
    for i in range(len(unique_label)):
        p_i = unique_label_count[i] / sum(unique_label_count)
        impurity -= p_i ** 2 
    return impurity
def information_gain( left, right, impurity): 
    left_idx=left.index
    right_idx=right.index
    p = float(len(left_idx)) / (len(left_idx) + len(right_idx))
    info_gain = impurity - ((p * gini_impurity(left, left_idx))- ((1 - p) * gini_impurity(right, right_idx)))
    return info_gain
def caclSE( dataSet):
    if (dataSet.shape [0] == 0): # If you enter a null data set, returns 0
        return 0
    return np.var (dataSet.iloc[:, -1]) * dataSet.shape [0] # * mean square variance = number of samples
 
def splitDataSet( dataSet, feature, value):

    dataSet = dataSet.fillna(1)
    arr1 = dataSet.iloc[np.nonzero(dataSet.iloc[:, feature].astype(int)  <= value) [0],:] # left
    arr2 = dataSet.iloc[np.nonzero(dataSet.iloc[:, feature].astype(int)  > value)[0], :] #right
    return arr1,arr2


def find_best_split(df, label):
    n = df.shape [1] 
    label_idx=label.index
    bestGain=0.0
    bestFeature = None
    bestValue = None
    minErr = np.inf
    df = df.loc[label_idx] # converting training data to pandas dataframe

    impurity = gini_impurity(label, label_idx) # determining the impurity at the current node    
    for col in range(n): 

        unique_values = set(df.iloc[:,col])
        unique_values ={x for x in unique_values  if x==x}
        for value in unique_values: 
            arr_L, arr_R = splitDataSet(df, col, value)

            IG=information_gain(arr_L,arr_R,impurity)

            if bestGain < IG:
                bestGain = IG
#                 bestFeature = df.iloc[:,col].name
                bestFeature = col
                bestValue = value
    return bestFeature,bestValue
        

# print(find_best_split(X_train, y_train))
def count(label, idx):
    unique_label, unique_label_counts = np.unique(label.loc[idx], return_counts=True)
    dict_label_count = dict(zip(unique_label, unique_label_counts))
    return dict_label_count
class Leaf:

    def __init__(self, label, idx):
        self.predictions = count(label, idx)
        
class Decision_Node:

    def __init__(self,
                 column,
                 value,
                 true_branch,
                 false_branch):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the col and value at this node
    print(spacing + f"[{node.column} <= {node.value}]")


    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
import sys 

sys.setrecursionlimit(10**6) 


def build_tree(df, label, max_depth=4): 

    idx=df.index
    best_col, best_value = find_best_split(df, label)
    if len (set (df.iloc[:, -1] .tolist ())) == 1: # If the current value of the same node, recursion End
        return Leaf(label, label.loc[idx].index)
    if max_depth==1:
        return Leaf(label, label.loc[idx].index)

    left_idx, right_idx = splitDataSet(df, best_col, best_value)
    
    true_branch = build_tree(left_idx, label, (max_depth-1))
    
    false_branch = build_tree(right_idx, label, (max_depth-1))
    
    return Decision_Node(best_col, best_value, true_branch, false_branch)
my_tree = build_tree(X_train, y_train,2)
def predict(test_data, tree):
    
    # Check if we are at a leaf node
    if isinstance(tree, Leaf): 
        return max(tree.predictions)
    
    # the current feature_name and value 
    feature_name, feature_value = tree.column, tree.value
    
    # pass the observation through the nodes recursively
    if test_data[feature_name] <= feature_value: 
        return predict(test_data, tree.true_branch)
    
    else: 
        return predict(test_data, tree.false_branch)
pred= X_test.apply(predict, axis=1, args=(my_tree,))

import graphviz
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(X_train, y_train)
 
sklearn_y_preds = dt.predict(X_test)
def accuracy(real,predicted):
    count=0
    real=np.array(real)
    predicted=np.array(predicted)
    for i in range(len(real)):
        if real[i]==predicted[i]:
            count+=1
    acc=count/len(real)
    return acc
print("Implementation:",accuracy(y_test,pred))
print("Sklearn:",accuracy(y_test,sklearn_y_preds))
print_tree(my_tree)

