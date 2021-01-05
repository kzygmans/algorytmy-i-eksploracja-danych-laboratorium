import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

global X_train, y_train
global X_test, y_test
global models, models_names, models_predict, models_predict_proba


def zadanie1():
    print('Zadanie 1')
    global X_train, y_train, X_test, y_test
    X_train = pd.read_csv("HAPT Data Set/Train/X_train.txt", header=None, delim_whitespace=True)
    y_train = pd.read_csv("HAPT Data Set/Train/y_train.txt", header=None, delim_whitespace=True)
    X_test = pd.read_csv("HAPT Data Set/Test/X_test.txt", header=None, delim_whitespace=True)
    y_test = pd.read_csv("HAPT Data Set/Test/y_test.txt", header=None, delim_whitespace=True)


def zadanie2():
    print('Zadanie 2')
    global models, models_names, models_predict, models_predict_proba
    print(type(X_train), type(y_train))
    print(X_train)
    print(y_train.values.ravel())

    # SVM
    X = X_train.copy()
    y = y_train.copy()
    y = y.values.ravel()
    print('after copy')
    print(X, y)
    clf_svm = svm.SVC(probability=True)
    clf_svm.fit(X, y)
    p1 = clf_svm.predict(X_test)
    pp1 = clf_svm.predict_proba(X_test)
    print("p1 result:")
    print(p1)
    print(y_test)

    # KNN
    X = X_train.copy()
    y = y_train.copy()
    y = y.values.ravel()
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X, y)
    p2 = clf_knn.predict(X_test)
    pp2 = clf_knn.predict_proba(X_test)
    print("p2 result:")
    print(p2)

    # Decision Tree
    X = X_train.copy()
    y = y_train.copy()
    y = y.values.ravel()
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt = clf_dt.fit(X, y)
    p3 = clf_dt.predict(X_test)
    pp3 = clf_dt.predict_proba(X_test)
    print("p3 result:")
    print(p3)

    # Random Forest
    X = X_train.copy()
    y = y_train.copy()
    y = y.values.ravel()
    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf.fit(X, y)
    p4 = clf_rf.predict(X_test)
    pp4 = clf_rf.predict_proba(X_test)
    print("p4 result:")
    print(p4)
    print(type(clf_svm), type(clf_knn))

    models_names = ['SVM', 'KNN', 'Decision Tree', 'Random Forest']
    models = [clf_svm, clf_knn, clf_dt, clf_rf]
    models_predict = [p1, p2, p3, p4]
    models_predict_proba = [pp1, pp2, pp3, pp4]



def zadanie3():
    print('Zadanie 3')
    X = X_test.copy()
    y = y_test.copy()
    y = y.values.ravel()

    # Confusion matrix
    print('Confusion matrix')
    for i in range(len(models_predict)):
        print('Confision matrix for model:', models_names[i])
        print(confusion_matrix(y, models_predict[i]))

    # ACC
    print('ACC')
    for i in range(len(models_predict)):
        acc = accuracy_score(y, models_predict[i])
        print('ACC for {} = {}'.format(models_names[i], acc))

    # Recall
    print('Recall')
    for i in range(len(models_predict)):
        recall = recall_score(y, models_predict[i], average='macro')
        print('Recall for {} = {}'.format(models_names[i], recall))

    # F1 score
    print('F1')
    for i in range(len(models_predict)):
        f1 = f1_score(y, models_predict[i], average='macro')
        print('F1 for {} = {}'.format(models_names[i], f1))

    # AUC
    print('AUC')
    for i in range(len(models_predict)):
        auc = roc_auc_score(y, models_predict_proba[i], multi_class='ovr', average='weighted')
        print('AUC for {} = {}'.format(models_names[i], auc))

    # Zadanie 3a
    # Kros-walidacja (CV)
    print('Kros-walidacja')
    for i in range(len(models)):
        print(models_names[i])
        score = cross_val_score(models[i], X, y, cv=5)
        print('Srednia:', score.mean())
        print('Odchylenie standardowe:', score.std() * 2)


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
