from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_frame = read_csv("D:/DZ/11sem/AI_Enregy/LR1/breast_cancer.csv");
data_frame.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
# print(data_frame);

numeral_val = get_dummies(data_frame);

# print(numeral_val);

