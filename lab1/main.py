from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def zadanie1():
    data = load_wine()

    print("Opis zbioru danych:")
    print(data.DESCR)
    print("Nazwy cech:")
    print(data.feature_names)
    print("Nazwy klas:")
    print(data.target_names)
    print("pandas DataFrame:")
    print(data.data)
    print("pandas Series:")
    print(data.target)


def zadanie2():
    data = load_wine(as_frame=True)

    X = data.frame.iloc[:, :-1]
    y = data.frame.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True)

    X_train.to_csv("out/X_train.csv")
    X_test.to_csv("out/X_test.csv")
    y_train.to_csv("out/y_train.csv")
    y_test.to_csv("out/y_test.csv")


def zadanie3():
    data = load_wine(as_frame=True)

    X = data.frame.iloc[:, :-1]
    X_train, X_test = train_test_split(X, train_size=0.6, shuffle=True)

    analiza("Zbior treningowy", X_train)
    analiza("Zbior testowy", X_test)


def analiza(dataset_name, dataset):
    print("Analiza ilosciowa dla: {}\n".format(dataset_name))
    for col in dataset.columns:
        print("Analiza dla kolumny: {}".format(col))
        print("Ilosc wartosci: {}".format(dataset[col].count()))
        print("Ilosc wartosci unikatowych: {}".format(dataset[col].nunique()))
        print("Wartosc srednia w zbiorze: {}".format(dataset[col].mean()))
        print("Ilosc wartosci null: {}".format(dataset[col].isnull().sum()))
        print("Wartosc maksymalna: {}".format(dataset[col].max()))
        print("Wartosc minimalna: {}".format(dataset[col].min()))
        print("Wartosc najczesciej wystepujaca w zbiorze: {}".format(dataset[col].mode().max()))
        print()


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
