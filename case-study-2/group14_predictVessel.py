import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
import functools
import os


def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc * 60 + x, map(int, hh_mm_ss.split(':')))


def make_features(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})

    time = df['SEQUENCE_DTTM'].to_numpy()
    lat = df['LAT'].to_numpy()
    lon = df['LON'].to_numpy()
    speed = df['SPEED_OVER_GROUND'].to_numpy()

    course_deg = df['COURSE_OVER_GROUND'].to_numpy() / 10.0
    course_rad = np.deg2rad(course_deg)

    course_sin = np.sin(course_rad)
    course_cos = np.cos(course_rad)

    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    lat_km = (lat - lat0) * 111.0
    lon_km = (lon - lon0) * 111.0 * np.cos(np.deg2rad(lat0))

    X = np.column_stack([
        time,
        lat_km,
        lon_km,
        speed,
        course_sin,
        course_cos
    ])

    X = preprocessing.StandardScaler().fit_transform(X)
    return X


def estimate_k(X):
    n = len(X)

    min_k = 8
    max_k = min(35, max(10, n // 5))

    best_k = 20
    best_score = -1

    for k in range(min_k, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=11111, n_init=20)
            labels = model.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k

        except:
            continue

    return best_k


def predictor_baseline(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})

    selected_features = [
        'SEQUENCE_DTTM',
        'LAT',
        'LON',
        'SPEED_OVER_GROUND',
        'COURSE_OVER_GROUND'
    ]

    X = df[selected_features].to_numpy()
    X = preprocessing.StandardScaler().fit(X).transform(X)

    K = 20
    model = KMeans(n_clusters=K, random_state=123, n_init=20).fit(X)
    labels_pred = model.predict(X)

    return labels_pred


def predictor(csv_path):
    X = make_features(csv_path)

    K = estimate_k(X)

    candidates = []

    try:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            random_state=123,
            n_init=10
        )
        labels_gmm = gmm.fit_predict(X)
        candidates.append(labels_gmm)
    except:
        pass

    try:
        kmeans = KMeans(
            n_clusters=K,
            random_state=123,
            n_init=50
        )
        labels_kmeans = kmeans.fit_predict(X)
        candidates.append(labels_kmeans)
    except:
        pass

    try:
        agg = AgglomerativeClustering(
            n_clusters=K,
            linkage='ward'
        )
        labels_agg = agg.fit_predict(X)
        candidates.append(labels_agg)
    except:
        pass

    best_labels = candidates[0]
    best_score = -1

    for labels in candidates:
        if len(np.unique(labels)) < 2:
            continue

        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
        except:
            continue

    return best_labels


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']

    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


if __name__ == "__main__":
    get_baseline_score()

    if os.path.exists('./Data/set3.csv'):
        evaluate()
    elif os.path.exists('./Data/set3noVID.csv'):
        labels_pred = predictor('./Data/set3noVID.csv')
        print(labels_pred)