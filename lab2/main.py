import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

global data

def zadanie1():
    global data
    col_names = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', nrows=0).columns
    data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', usecols=[col for col in col_names if 'Product_Code' in col or 'Normalized' in col])
    print(col_names)
    print(data)

def zadanie2():
    # Parameters
    number_of_clusters = 5

    # KMeans
    dataset = data.drop(data.iloc[:, :1], inplace=False, axis=1)
    kMeans = KMeans(n_clusters=number_of_clusters, init='random')
    kMeans.fit(dataset)
    pca_2d = PCA(n_components=2)
    pca_data = pd.DataFrame(pca_2d.fit_transform(dataset))
    print(pca_data)
    dataset['cluster'] = kMeans.labels_
    pca_data['cluster'] = kMeans.labels_
    centroids = pd.DataFrame(pca_2d.fit_transform(kMeans.cluster_centers_))
    subplot1 = centroids.plot.scatter(0, 1, marker='*', s=150, linewidths=2, color='red', zorder=8, label='Centoridy')
    markers = ['o', 'v', 's', 'h', 'p']
    colors = ['goldenrod', 'yellowgreen', 'skyblue', 'plum', 'silver']
    for i in range(number_of_clusters):
        v = pca_data.loc[pca_data['cluster'] == i]
        subplot1.scatter(v[0], v[1], color=colors[i], marker=markers[i], label='Skupienie ' + str(i))
    subplot1.legend(loc='upper right')
    subplot1.grid(True)
    plt.title("KMeans")
    plt.savefig("plot_z2_1.png")

    # KMeans++
    dataset2 = data.drop(data.iloc[:, :1], inplace=False, axis=1)
    kMeansPP = KMeans(n_clusters=number_of_clusters, init='k-means++')
    kMeansPP.fit(dataset2)
    pca_2d = PCA(n_components=2)
    pca_data = pd.DataFrame(pca_2d.fit_transform(dataset2))
    print(pca_data)
    dataset2['cluster'] = kMeansPP.labels_
    pca_data['cluster'] = kMeansPP.labels_
    print(dataset2)
    print(pca_data)
    centroids = pd.DataFrame(pca_2d.fit_transform(kMeansPP.cluster_centers_))
    subplot1 = centroids.plot.scatter(0, 1, marker='*', s=150, linewidths=2, color='red', zorder=8, label='Centoridy')
    markers = ['o', 'v', 's', 'h', 'p']
    colors = ['goldenrod', 'yellowgreen', 'skyblue', 'plum', 'silver']
    for i in range(number_of_clusters):
        v = pca_data.loc[pca_data['cluster'] == i]
        subplot1.scatter(v[0], v[1], color=colors[i], marker=markers[i], label='Skupienie ' + str(i))
    subplot1.legend(loc='upper right')
    subplot1.grid(True)
    plt.title("KMeans++")
    plt.savefig("plot_z2_2.png")

    plt.show()

def zadanie3():
    # Parameters
    metric = ''

if __name__ == "__main__":
    zadanie1()
    zadanie2()

