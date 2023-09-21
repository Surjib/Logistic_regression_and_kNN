from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_frame = read_csv("D:/DZ/11sem/AI_Enregy/LR1/breast_cancer.csv");
data_frame.drop(["Unnamed: 32"], axis = 1, inplace = True)
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


k_values = []
for i in range(50):
    k_values.append(i+1)

knn_results = []
for k in range (len(k_values)):
    knn_model = KNeighborsClassifier(n_neighbors= k+1)
    knn_model.fit(X_train, Y_train)
    Y_pred = knn_model.predict(X_test.values)
    knn_results.append(accuracy_score(Y_test, Y_pred))

print(knn_results)

plt.plot(k_values, knn_results, 'blue')
# plt.title(title, fontsize=10)
plt.ylabel('Accuracy', fontsize=8)
plt.xlabel('№ of Neighbours', fontsize=8)
plt.grid(True)
plt.show()


