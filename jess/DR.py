import matplotlib.pyplot as plt  # 画图工具
from sklearn import datasets                       # 数据集导入
from sklearn.decomposition import PCA              # 导入降维方法PCA
from sklearn.manifold import TSNE                  # 导入降维方法T-SNE
from sklearn.cluster import KMeans                 # 导入聚类方法K-means
from sklearn.metrics import fowlkes_mallows_score  # FMI评价法
from sklearn.metrics import accuracy_score         # 准确率评价
from sklearn import metrics                        # CH指数评价
from sklearn.metrics import davies_bouldin_score   # DBI指数评价
from sklearn.preprocessing import StandardScaler   # 标准差标准化数据

from sklearn.preprocessing import MinMaxScaler     # 最大最小值标准化

# 数据集导入
wine = datasets.load_wine()
# 数据集分割成数据和标签
wine_data = wine['data']
wine_target = wine['target']
# 标准差标准化数据
wine_data = StandardScaler().fit(wine_data).transform(wine_data)
wine_data = MinMaxScaler().fit(wine_data).transform(wine_data)
print('数据集的长度', len(wine_data))
print('数据集中数据的类型', wine_data.dtype)
print('数据集的类型', type(wine_data))
# print('数据集的数据', wine_data)
print('原数据集样本的标签\n', wine_target)

wine_name = wine['feature_names']
print('数据集的特征名', wine_name)
wine_desc = wine['DESCR']
print('数据集的描述信息', wine_desc)
print('原始数据集数据的形状为：', wine_data.shape)
# print('原始数据集标签的形状为：', wine_target.shape)
# 降维前聚类
kmeans = KMeans(n_clusters=3,random_state=10).fit(wine_data)
print(kmeans.labels_)
score = fowlkes_mallows_score(wine_target, kmeans.labels_)
print("降维前聚类wine数据集的FMI:%f" % (score))
print('降维前聚类各样本的标签\n',kmeans.labels_)
print('准确率：',accuracy_score(kmeans.labels_,wine_target))
# 利用PCA、TSNE降维，降到二维
pca = PCA(n_components=2)
P_data = pca.fit(wine_data).transform(wine_data)
print('降维后数据集的形状', P_data.shape)
tsne = TSNE(n_components=2)
T_data = tsne.fit_transform(wine_data)
print('降维后数据集的形状', T_data.shape)
# 采用降维后的数据进行聚类（K-means）
kmeans_P = KMeans(n_clusters=3,random_state=10).fit(P_data)
kmeans_T = KMeans(n_clusters=3, random_state=10).fit(T_data)
print('构建的KMeans模型为：', kmeans)
# print(kmeans.labels_)
score_P = fowlkes_mallows_score(wine_target, kmeans_P.labels_)
score_T = fowlkes_mallows_score(wine_target, kmeans_T.labels_)
print("PCA降维后聚类wine数据集的FMI:%f" % (score_P))
print("TSNE降维后聚类wine数据集的FMI:%f" % (score_T))
print('PCA降维后各样本聚类的标签\n',kmeans_P.labels_)
print('TSNE降维后各样本聚类的标签\n',kmeans_T.labels_)
arr_P = accuracy_score(kmeans_P.labels_,wine_target)
print('PCA降维后准确率：%f'%(arr_P))
arr_T = accuracy_score(kmeans_T.labels_,wine_target)
print('T-SNE降维后准确率：%f'%(arr_T))
# Calinski-Harabaz 指数

CH_P=metrics.calinski_harabasz_score(wine_data, kmeans_P.labels_)
print('PCA降维后CH指数：%f' % (CH_P))
CH_T=metrics.calinski_harabasz_score(wine_data, kmeans_T.labels_)
print('T-SNE降维后CH指数：%f' % (CH_T))

# Davies-Bouldin Index

DBI_P = davies_bouldin_score(wine_data, kmeans_P.labels_)
print('PCA降维后DBI指数：%f' % (DBI_P))
DBI_T = davies_bouldin_score(wine_data, kmeans_T.labels_)
print('T-SNE降维后DBI指数：%f' % (DBI_T))

# CH_P=metrics.calinski_harabasz_score(wine_target, kmeans_P.labels_)
print(CH_P)
for i in range(0,100):
    kmeans_p = KMeans(n_clusters = 3,random_state =i).fit(P_data)
    kmeans_t = KMeans(n_clusters = 3,random_state =i).fit(T_data)
    print('准确率：',i,accuracy_score(kmeans_p.labels_,wine_target),accuracy_score(kmeans_t.labels_,wine_target))

