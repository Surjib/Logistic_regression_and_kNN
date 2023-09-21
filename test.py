import numpy as np
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data_frame = read_csv("D:/DZ/11sem/AI_Enregy/LR1/breast_cancer.csv");
data_frame.drop(["id",  "Unnamed: 32"], axis = 1, inplace = True)
# print(data_frame);




# data_frame["Zero"] = 0;
#
# Y = data_frame["Zero"];
# print(X, "\n----------------------------");
# print(Y);
numeral_val = get_dummies(data_frame);
# print(numeral_val);
diagnosis = numeral_val.loc[:,"diagnosis_B"];

# print(diagnosis)
X_train, X_test, Y_train, Y_test = train_test_split(numeral_val, diagnosis, test_size=0.2, random_state=42);

# print(X_train, X_test, Y_train, Y_test);

# print(numeral_val);

# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train, Y_train)
kf = KFold(n_splits=5, shuffle=True);

k_values = []
for i in range(50):
    k_values.append(i+1)

knn_results_test = []
knn_results_train = []

cross_score_train = []
cross_score_test = []


for k in range (len(k_values)):
    knn_model = KNeighborsClassifier(n_neighbors= k+1)

    knn_model.fit(X_train, Y_train)
    Y_pred = knn_model.predict(X_test.values)
    knn_results_train.append(accuracy_score(Y_test, Y_pred) * 100)

    cross_score_train.append(cross_val_score(knn_model, X_train.values, Y_train.values, cv=kf, scoring='accuracy'))

    knn_model.fit(X_test, Y_test)
    Y_pred = knn_model.predict(X_test.values)
    knn_results_test.append(accuracy_score(Y_test, Y_pred) * 100)

    cross_score_test.append(cross_val_score(knn_model, X_test.values, Y_test.values, cv=kf, scoring='accuracy'))


# plt.plot(k_values, knn_results, 'blue')
# # plt.title(title, fontsize=10)
# plt.ylabel('Accuracy', fontsize=8)
# plt.xlabel('№ of Neighbours', fontsize=8)
# plt.grid(True)
# plt.show()

plt.subplot(1, 2, 1)
plt.plot(k_values, knn_results_train, 'blue')
plt.title("Train", fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=8)
plt.xlabel('Neighbours', fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, knn_results_test, 'blue')
plt.title("Test", fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=8)
plt.xlabel('Neighbours', fontsize=8)
plt.grid(True)


plt.show()



C = np.arange(0.01,1,0.01)
LogisticRegression(C=C)

