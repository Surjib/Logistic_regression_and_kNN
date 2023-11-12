import numpy as np
import pandas as pd
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def check_results_KNN (k_values, X_train, X_test, Y_train, Y_test):
    KNN_model_results_test = []
    KNN_model_results_train = []
    KNN_cross_score_train = []
    KNN_cross_score_test = []

    for k in range (len(k_values)):
        knn_model = KNeighborsClassifier(n_neighbors= k+1)

        knn_model.fit(X_train.values, Y_train.values)
        Y_pred = knn_model.predict(X_test.values)
        KNN_model_results_train.append(accuracy_score(Y_test.values, Y_pred) * 100)

        KNN_cross_score_train.append(cross_val_score(knn_model, X_train.values, Y_train.values, cv=kf, scoring='accuracy'))

        knn_model.fit(X_test.values, Y_test.values)
        Y_pred = knn_model.predict(X_test.values)
        KNN_model_results_test.append(accuracy_score(Y_test.values, Y_pred) * 100)

        KNN_cross_score_test.append(cross_val_score(knn_model, X_test.values, Y_test.values, cv=kf, scoring='accuracy'))
    return  KNN_model_results_train, KNN_cross_score_train, KNN_model_results_test, KNN_cross_score_test


def rescaler (x_values, diagnosis):
    scaler = StandardScaler()
    X_new = scaler.fit_transform(x_values)
    print(X_new)

    X_train_res, X_test_res, Y_train_res, Y_test_res = train_test_split(X_new, diagnosis, test_size=0.2, random_state=42);
    return X_train_res, X_test_res, Y_train_res, Y_test_res

def check_results_LR(C, X_train, X_test, Y_train, Y_test):
    LR_model_results_test = []
    LR_model_results_train = []
    LR_cross_score_train = []
    LR_cross_score_test = []

    for j in C:
        lr_model = LogisticRegression(C=j, solver='lbfgs', max_iter=100)

        lr_model.fit(X_train.values, Y_train.values)
        Y_pred = lr_model.predict(X_test.values)
        LR_model_results_train.append(accuracy_score(Y_test.values, Y_pred) * 100)
        LR_cross_score_train.append(cross_val_score(lr_model, X_train.values, Y_train.values, cv=kf, scoring='accuracy'))

        lr_model.fit(X_train.values, Y_train.values)
        Y_pred = lr_model.predict(X_test.values)
        LR_model_results_test.append(accuracy_score(Y_test.values, Y_pred) * 100)
        LR_cross_score_test.append(cross_val_score(lr_model, X_test.values, Y_test.values, cv=kf, scoring='accuracy'))

    return LR_model_results_train, LR_cross_score_train, LR_model_results_test, LR_cross_score_test


def result_graphs(X, model_test, model_train, score_test, score_train, is_knn):
    if is_knn:
        label = 'Neighbours'
    else:
        label = 'C'

    plt.subplot(2, 2, 1)
    plt.plot(X, model_train, 'blue')
    plt.title("Train", fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=8)
    plt.xlabel(label, fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(X, model_test, 'blue')
    plt.title("Test", fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=8)
    plt.xlabel(label, fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(X, score_train)
    plt.title("Train", fontsize=10)
    plt.ylabel('CV score', fontsize=8)
    plt.xlabel(label, fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(X, score_test)
    plt.title("Test", fontsize=10)
    plt.ylabel('CV score', fontsize=8)
    plt.xlabel(label, fontsize=8)
    plt.grid(True)


data_frame = read_csv("D:/DZ/11sem/AI_Enregy/LR1/breast_cancer.csv");
# print(data_frame);
data_frame.drop(["id",  "Unnamed: 32"], axis = 1, inplace = True)


numeral_val = get_dummies(data_frame);
diagnosis = numeral_val.loc[:,"diagnosis_B"];

kf = KFold(n_splits=5, shuffle=True);


k_values = []

X_train, X_test, Y_train, Y_test = train_test_split(numeral_val, diagnosis, test_size=0.2, random_state=42);




for i in range(50):
    k_values.append(i+1)

KNN_model_results_train, KNN_cross_score_train, KNN_model_results_test, KNN_cross_score_test = check_results_KNN(k_values, X_train, X_test, Y_train, Y_test)
result_graphs(k_values, KNN_model_results_test, KNN_model_results_train, KNN_cross_score_test, KNN_cross_score_train, True)

plt.show()


C = np.arange(0.01,1,0.01)

LR_model_results_train, LR_cross_score_train, LR_model_results_test, LR_cross_score_test = check_results_LR(C, X_train, X_test, Y_train, Y_test)
result_graphs(C, LR_model_results_test, LR_model_results_train, LR_cross_score_test, LR_cross_score_train, False)

plt.show()



X_train_res, X_test_res, Y_train_res, Y_test_res = rescaler(numeral_val, diagnosis)

X_train_res_pd = pd.DataFrame(X_train_res)
X_test_res_pd = pd.DataFrame(X_test_res)
Y_train_res_pd = pd.DataFrame(Y_train_res)
Y_test_res_pd = pd.DataFrame(Y_test_res)


KNN_model_results_train_res, KNN_cross_score_train_res, KNN_model_results_test_res, KNN_cross_score_test_res = check_results_KNN(k_values, X_train_res_pd, X_test_res_pd, Y_train_res_pd, Y_test_res_pd)
result_graphs(k_values, KNN_model_results_test_res, KNN_model_results_train_res, KNN_cross_score_test_res, KNN_cross_score_train_res, True)

plt.show()


LR_model_results_train_res, LR_cross_score_train_res, LR_model_results_test_res, LR_cross_score_test_res = check_results_LR(C, X_train_res_pd, X_test_res_pd, Y_train_res_pd, Y_test_res_pd)
result_graphs(C, LR_model_results_test_res, LR_model_results_train_res, LR_cross_score_test_res, LR_cross_score_train_res, False)


plt.show()

