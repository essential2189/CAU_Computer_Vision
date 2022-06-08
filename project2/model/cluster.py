from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def knn_cluster(result_list, label_list, normal_list, anomaly_list):
    scaler1 = StandardScaler()
    pca1 = PCA(n_components=500)

    train_scaler = scaler1.fit_transform(result_list)
    train_reduce = pca1.fit_transform(train_scaler)

    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

    knn.fit(train_reduce, label_list)

    y_all = knn.predict(train_reduce).tolist()

    return roc_auc_score(label_list, y_all)