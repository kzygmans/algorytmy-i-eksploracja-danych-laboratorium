import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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
    subplot2 = centroids.plot.scatter(0, 1, marker='*', s=150, linewidths=2, color='red', zorder=8, label='Centoridy')
    markers = ['o', 'v', 's', 'h', 'p']
    colors = ['goldenrod', 'yellowgreen', 'skyblue', 'plum', 'silver']
    for i in range(number_of_clusters):
        v = pca_data.loc[pca_data['cluster'] == i]
        subplot2.scatter(v[0], v[1], color=colors[i], marker=markers[i], label='Skupienie ' + str(i))
    subplot2.legend(loc='upper right')
    subplot2.grid(True)
    plt.title("KMeans++")
    plt.savefig("plot_z2_2.png")


def zadanie3():
    # Parameters
    metric = 'euclidean'

    # AgglomerativeClustering
    dataset3 = data.drop(data.iloc[:, :1], inplace=False, axis=1)
    agglomerative = AgglomerativeClustering(n_clusters=5, affinity=metric, linkage='ward')
    agglomerative.fit(dataset3)
    print(agglomerative)
    print(agglomerative.labels_)
    dataset3['cluster'] = agglomerative.labels_
    print(dataset3)
    g = sns.clustermap(dataset3, metric=metric)
    g.savefig("plot_z3_1.png")

    # DBSCAN
    dataset4 = data.drop(data.iloc[:, :1], inplace=False, axis=1)
    pca_2d = PCA(n_components=2)
    pca_data4 = pd.DataFrame(pca_2d.fit_transform(dataset4))
    dbscan = DBSCAN(metric=metric, eps=0.15, min_samples=10)
    dbscan.fit(pca_data4)
    print(dbscan.labels_)
    pca_data4['cluster'] = dbscan.labels_
    subplot4 = plt.subplot()
    c1 = pca_data4.loc[pca_data4['cluster'] == 0]
    subplot4.scatter(c1[0], c1[1], color='blue', marker='*', label='Skupienie 1')
    c2 = pca_data4.loc[pca_data4['cluster'] == 1]
    subplot4.scatter(c2[0], c2[1], color='red', marker='^', label='Skupienie 2')
    subplot4.legend(loc='upper right')
    subplot4.grid(True)
    plt.savefig("plot_z3_2.png")
    plt.show()


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
