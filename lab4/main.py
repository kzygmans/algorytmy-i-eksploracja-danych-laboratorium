import time
from openpyxl import Workbook

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

global X_train, y_train
global X_test, y_test


def zadanie1():
    print('Zadanie 1')
    global X_train, y_train, X_test, y_test
    X_train = pd.read_csv("HAPT Data Set/Train/X_train.txt", header=None, delim_whitespace=True)
    y_train = pd.read_csv("HAPT Data Set/Train/y_train.txt", header=None, delim_whitespace=True)
    X_test = pd.read_csv("HAPT Data Set/Test/X_test.txt", header=None, delim_whitespace=True)
    y_test = pd.read_csv("HAPT Data Set/Test/y_test.txt", header=None, delim_whitespace=True)


def zadanie2():
    print('Zadanie 2')
    # Test bez redukcji wielowymiarowosci
    start = time.process_time()
    clf_svm = svm.SVC(probability=True)
    clf_svm.fit(X_train, y_train.values.ravel())
    train_time = time.process_time() - start
    print("Train time: {}".format(train_time))
    start = time.process_time()
    score = cross_val_score(clf_svm, X_test, y_test.values.ravel(), cv=5)
    test_time = time.process_time() - start
    mean_score = score.mean()
    print("Test time: {}".format(test_time))
    print("Score: {}".format(mean_score))

    # Test z redukcja wielowymiarowosci
    pca = PCA(n_components=10)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    print(X_train_pca)
    start = time.process_time()
    clf_svm = svm.SVC(probability=True)
    clf_svm.fit(X_train_pca, y_train.values.ravel())
    train_time_pca = time.process_time() - start
    print("Train time PCA: {}".format(train_time_pca))
    X_test_pca = pd.DataFrame(pca.fit_transform(X_test))
    start = time.process_time()
    score_pca = cross_val_score(clf_svm, X_test_pca, y_test.values.ravel(), cv=5)
    test_time_pca = time.process_time() - start
    mean_score_pca = score_pca.mean()
    print("Test time PCA: {}".format(test_time_pca))
    print("Score PCA: {}".format(mean_score_pca))

    wb = Workbook()
    sheet = wb.active
    sheet['B1'] = "Base dataset"
    sheet['C1'] = "After PCA reduction"
    sheet['A2'] = "Train time"
    sheet['B2'] = train_time
    sheet['C2'] = train_time_pca
    sheet['A3'] = "Test time"
    sheet['B3'] = test_time
    sheet['C3'] = test_time_pca
    sheet['A4'] = "Score"
    sheet['B4'] = mean_score
    sheet['C4'] = mean_score_pca
    wb.save('dim_reduction.xlsx')


def zadanie3():
    print('Zadanie 3')
    clf1 = svm.SVC(C=0.9, probability=True)
    clf2 = KNeighborsClassifier(n_neighbors=13)
    clf3 = tree.DecisionTreeClassifier(min_samples_split=5)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=80)
    ensemble = VotingClassifier(estimators=[("SVM", clf1),
                                            ("KNN", clf2),
                                            ("DecisionTree", clf3),
                                            ("RandomForest", clf4)],
                                voting='soft', weights=[0.3, 0.3, 0.2, 0.2])
    ensemble.fit(X_train, y_train.values.ravel())
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)

    acc = accuracy_score(y_test.values.ravel(), y_pred)
    print("ACC = {}".format(acc))
    recall = recall_score(y_test.values.ravel(), y_pred, average='macro')
    print("Recall = {}".format(recall))
    f1 = f1_score(y_test.values.ravel(), y_pred, average='macro')
    print("F1 = {}".format(f1))
    auc = roc_auc_score(y_test.values.ravel(), y_pred_proba, multi_class='ovr', average='weighted')
    print("AUC = {}".format(auc))

    wb = Workbook()
    sheet = wb.active
    sheet['A1'] = "Ensambled learning results"
    sheet['A2'] = "ACC"
    sheet['B2'] = acc
    sheet['A3'] = "Recall"
    sheet['B3'] = recall
    sheet['A4'] = "F1"
    sheet['B4'] = f1
    sheet['A5'] = "AUC"
    sheet['B5'] = auc
    wb.save('ensambled_learning.xlsx')


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
