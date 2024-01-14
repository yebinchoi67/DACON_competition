1주차 스터디 - 관련 모델 및 알고리즘 탐색
===
---

# **<주제>**

**신용카드 사기 거래 탐지 AI 경진대회**

비식별화된 신용카드 거래 데이터로부터 사기 거래를 탐지하는 AI 솔루션 개발

# **<할 일>**

unsupervised classification,

unsupervised anomaly detection

위 키워드로 어떤 모델, 알고리즘이 있는지

### <cf. 관련 용어 정리>

**비지도 학습:** 학습 훈련 데이터(training data)로 출력 없이 입력만 제공되는 상황. 문제(입력)의 답(출력)을 가르쳐 주지 않는 것

**unsuperivised classification(비지도 분류)**

비지도 학습에는 Clustering(군집화), Association(연관)이 쓰임.

**Association**: 컬렉션에서 항목이 동시에 발생할 확률을 발견하는 규칙 기반 머신러닝.

**Clustering:** 객체를 유사한 클러스터로 나누고 다른 클러스터에 속하는 객체와 유사하지 않은 클러스터로 나누는 방법

https://blog.naver.com/sdssoft/222575331310

https://developer.ibm.com/articles/cc-unsupervised-learning-data-classification/

계층적 군집분석(Hierarchical Clustering)과 비계층적 군집분석(Non-Hierarchical Clustering) 분석으로 나뉨.

**비계층적 군집분석(Non-Hierarchical Clustering)**

1. **중심 기반(Center-based) : K-means**
K-means는 중심기반(Center-based) 클러스터링 방법으로 “유사한 데이터는 중심점(centroid)을 기반으로 분포할 것이다”는 가정을 기반으로 한다.
n개의 데이터와 k(<=n)개의 중심점(centroid)이 주어졌을때 각 그룹 내의 데이터와 중심점 간의 비용(거리)을 최소화하는 방향으로 계속 업데이트를 해줌으로써 그룹화를 수행하는 기법이다.
    1. 초기점(k) 설정
        
        k는 중심점(centroid)이자, 묶일 그룹(cluster)의 수와 같다.
        
        위 예시에서는 k=3으로 설정(동그라미)
        
    2. 그룹(cluster) 부여
        
        k개의 중심점(동그라미)과 개별 데이터(네모)간의 거리를 측정한다.
        
        가장 가까운 중심점으로 데이터를 부여한다.
        
    3. 중심점(centroid) 업데이트
        
        할당된 데이터들의 평균값(mean)으로 새로운 중심점(centroid)을 업데이트한다.
        
    4. 최적화
    
    2,3번 작업을 반복적으로 수행한다.
    
    변화가 없으면 작업을 중단한다.
    
    결국 아래와 같은 목적함수를 최소화하는 것을 목표로 하는 알고리즘인 것이다
    
   ![image01.png](https://github.com/yebinchoi67/DACON_competition/blob/b176accacff7b2556cf2eb1d8c5d440e13dc44bc/Credit_card_fraud_detection/images/image01.png)
    
2. **밀도 기반(Density-based) : DBSCAN**
DBSCAN는 밀도기반(Density-based) 클러스터링 방법으로 “유사한 데이터는 서로 근접하게 분포할 것이다”는 가정을 기반으로 한다. K-means와 달리 처음에 그룹의 수(k)를 설정하지 않고 자동적으로 최적의 그룹 수를 찾아나간다.
    
    ![image02.png](https://github.com/yebinchoi67/DACON_competition/blob/20c4787fd5f02b85a802b23ab430d4c33ae77cdb/Credit_card_fraud_detection/images/image02.png)
    
    먼저 하나의 점(파란색)을 중심으로 반경(eps) 내에 최소 점이 4개(minPts=4)이상 있으면, 하나의 군집으로 판단하며 해당 점(파란색)은 Core가 된다.
    
    반경 내에 점이 3개 뿐이므로 Core가 되진 않지만 Core1의 군집에 포함된 점으로, 이는 Border가 된다.
    
    1번과 마찬가지로 Core가 된다.
    
    그런데 반경내의 점중에 Core1이 포함되어 있어 군집을 연결하여 하나의 군집으로 묶인다.
    
    이와 같은 방식으로 군집의 확산을 반복하면서, 자동으로 최적의 군집수가 도출된다.
    

**unsupervised anomaly detection (비지도 이상탐지)**

https://towardsdatascience.com/unsupervised-anomaly-detection-in-python-f2e61be17c2b

**anomaly detection(이상치 탐지란)?**

데이터 세트에서 나머지와 다른 데이터 포인트를 찾는 과정. 즉, 데이터 중에서 일반적인 방법이 아닌 방법으로 생성된 데이터를 검출하는 것을 의미 (ex. 이상탐지에는 금융거래의 사기 탐지, 오류 탐지 및 예측 유지 관리가 포함됨.)

**unsupervised anomaly detection(비지도 이상탐지)란?**

unsupervised anomaly detection(비지도 이상 탐지)는 라벨이 없는 데이터 세트를 포함한다.(즉, 정상, 비정상이 표시되지 않은 데이터 세트) 라벨이 지정되지 않은 데이터 집합의 major한 데이터 점이 "정상"이라고 가정하고 "정상" 데이터 지점과 다른 데이터 지점을 찾아 그 데이터 점을 이상치로 간주한다.

**unsupervised anomaly detection 알고리즘?모델? 종류**

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173

**k-NN Global Anomaly Detection**

The k-nearest-neighbor global unsupervised anomaly detection algorithm is a straightforward way for detecting anomalies and not to be confused with k-nearest neighbor classification. As the name already implies, it focuses on global anomalies and is not able to detect local anomalies. First, for every record in the dataset, the k-nearest-neighbors have to be found. Then, an anomaly score is computed using these neighbors, whereas two possibilities have been proposed: Either the distance to the kth-nearest-neighbor is used (a single one) or the average distance to all of the k-nearest-neighbors is computed. In the following, we refer to the first method as kth-NN and the latter as k-NN. In practical applications, the k-NN method is often preferred. However, the absolute value of the score depends very much on the dataset itself, the number of dimensions, and on normalization. As a result, it is in practice not easy to select an appropriate threshold, if required.

The choice of the parameter k is of course important for the results. If it is chosen too low, the density estimation for the records might be not reliable. On the other hand, if it is too large, density estimation may be too coarse. As a rule of thumb, k should be in the range 10 < k < 50. In classification, it is possible to determine a suitable k, for example by using cross-validation. Unfortunately, there is no such technique in unsupervised anomaly detection due to missing labels. For that reason, we use later in the evaluation many different values for k and average in order to get a fair evaluation when comparing algorithms.

In Fig 4 we exemplary illustrate how the result of an unsupervised anomaly detection algorithm (here: k-NN with k = 10) can be visualized. The plot was generated using a simple, artificially generated two-dimensional dataset with four Gaussian clusters and uniformly sampled anomalies. After applying the global k-NN, the outlier scores are visualized by the bubble-size of the corresponding instance. The color indicates the label, whereas anomalies are red. It can be seen, that k-NN cannot detect the anomalies close to the clusters well and assign small scores.

**Local outlier factor (LOF)**

Local outlier factor is probably the most common technique for anomaly detection. This algorithm is based on the concept of the local density. It compares the local density of an object with that of its neighbouring data points. If a data point has a lower density than its neighbours, then it is considered an outlier.

https://godongyoung.github.io/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2019/03/11/Local-Outlier-Factor(LOF).html

https://en.wikipedia.org/wiki/Local_outlier_factor

**Connectivity-Based Outlier Factor (COF) - LOF랑 비슷한데 밀도 추정이 다름?**

The connectivity-based outlier factor is similar to LOF, but the density estimation for the records is performed differently. In LOF, the k-nearest-neighbors are selected based on the Euclidean distance. This indirectly assumes, that the data is distributed in a spherical way around the instance. If this assumption is violated, for example if features have a direct linear correlation, the density estimation is incorrect. COF wants to compensate this shortcoming and estimates the local density of the neighborhood using an shortest-path approach, called the chaining distance. Mathematically, this chaining distance is the minimum of the sum of all distances connecting all k neighbors and the instance. For simple examples, where features are obviously correlated, this density estimation approach performs much more accurate. Fig 5 shows the outcome for LOF and COF in direct comparison for a simple two-dimensional dataset, where the attributes have a linear dependency. It can be seen that the spherical density estimation of LOF cannot detect the outlier, but COF succeeded by connecting the normal records with each other for estimating the local density.

**Influenced Outlierness (INFLO)**

When a dataset contains clusters with different densities and they are close to each other, it can be shown that LOF fails scoring the instances at the borders of the clusters correctly. The influenced outlierness (INFLO) algorithm uses besides the k-nearest-neighbors additionally a reverse nearest neighborhood set, in which records are stored for with the current record is a neighbor. For computing the INFLO score, both neighborhood sets are combined. Then, the local density of this set and the score is computed the same way as for LOF. This procedure is illustrated in Fig 6, where for the red instance the 6-nearest-neighbors reside in the gray area. This red instance will clearly be detected as an anomaly by LOF, since five of its neighbors have a much higher local density. For INFLO, also the instances are taken into account for which the red instance is a neighbor (the blue instances). Using this extended set, the red instance is less likely to be detected as an anomaly by INFLO. Please note, that the set of k-nearest-neighbors typically contains k instances (with the exception of ties), whereas the reverse nearest neighborhood set may contain any amount. Depending on the data, it might contain no instance, exactly k or even more instances. When using this strategy, it is possible to compute more accurate anomaly scores when clusters of different densities are close to each other.

**Local Outlier Probability (LoOP)**

Until now, all presented algorithms output anomaly scores, which are more handy than binary labels. When comparing the global k-NN algorithm and LOF, the property of having a reference point for normal instances of LOF seems even better than the arbitrary score of k-NN. Unfortunately, it is still not clear in LOF, above which score threshold we can clearly think about an anomaly. The local outlier probability (LoOP) [46] tries to address this issue by outputting an anomaly probability instead of a score, which might also result in better comparison of anomalous records between different datasets.

Similar to the previous local algorithms, LoOP also uses a neighborhood set for local density estimation. In contrast to other algorithms, it computes this density differently: The basic assumption is that the distances to the nearest neighbors follow a Gaussian distribution. Since distances are always positive, LoOP assumes a “half-Gaussian” distribution and uses its standard deviation, called the probabilistic set distance. It is used (similar to LOF) as a local density estimation—the ratios of each instance compared to its neighbors results in a local anomaly detection score. For converting this score into a probability, a normalization and a Gaussian error function is applied finally. The idea of having a probabilistic output instead of a score is very useful. However, some critical thoughts should arise in this context [29]. For example, if the algorithm assigns a 100% probability to an instance, what would happen, if we add another instance to the dataset which is more anomalous then that? As we can see from this simple example, probabilities are still relative to the data and might not differ too much from a normalized score.

**Local Correlation Integral (LOCI)**

For all of the above algorithms, choosing k is a crucial decision for detection performance. As already mentioned, there is no way of estimating a good k based on the data. Nevertheless, the local correlation integral (LOCI) [47] algorithm addresses this issue by using a maximization approach. The basic idea is that all possible values of k are used for each record and finally the maximum score is taken. To achieve this goal, LOCI defines the r-neighborhood by using a radius r, which is expanded over time. Similar to LoOP, the local density is also estimated by using a half-Gaussian distribution, but here the amount of records in the neighborhood is used instead of the distances. Also, the local density estimation is different in LOCI: It compares two different sized neighborhoods instead of the ratio of the local densities. A parameter α controls the ratio of the different neighborhoods. Removing the critical parameter k comes at a price. Typically, nearest-neighbor based anomaly detection algorithms have computational complexity of O(n2) for finding the nearest neighbors. Since in LOCI additionally the radius r needs to be expanded from one instance to the furthest, the complexity increases to O(n3), which makes LOCI too slow for larger datasets.

**Approximate Local Correlation Integral (aLOCI)**

The authors of LOCI were aware of the long runtime and proposed aLOCI [48], a faster but approximate version of LOCI. aLOCI uses quad trees to speed up the counting of the two neighborhoods using some constraints for α. If a record is in the center of a cell of such a quad tree, the counting estimation is good, but if it is at the border, the approximation might be bad. For that reason, multiple (g) quad trees are constructed with the hope, that there is a good approximative tree for every instance. Furthermore, the tree depth (L) needs to be specified. The authors claim that the total complexity of their algorithm, comprising of tree creation and scoring, is O(NLdg + NL(dg + 2d)), whereas d is the number of dimensions. As typical for tree approaches, it can be seen that the number of dimensions has a very negative impact on the runtime. During our evaluation, we experienced very different results from aLOCI. Sometimes results seem reasonable and sometimes results showed a very poor anomaly detection performance. This observation was tracked down the tree creation process. For a perfect estimation, N trees are required. Since the trick of this algorithm is to use only g trees, this also turned out to be a weak point: If, by chance, the trees represented the normal instances well, many approximations were correct and thus the output of the algorithm. On the other hand, if the trees did not well represent the majority of the data, the anomaly detection performance was unacceptable.

**Cluster-Based Local Outlier Factor (CBLOF/ uCBLOF)**

All previous anomaly detection algorithms are based on density estimation using nearest-neighbors. In contrast, the cluster-based local outlier factor (CBLOF) [49] uses clustering in order to determine dense areas in the data and performs a density estimation for each cluster afterwards. In theory, every clustering algorithm can be used to cluster the data in a first step. However, in practice k-means is commonly used to take advantage of the low computational complexity, which is linear compared to the quadratic complexity of the nearest-neighbor search. After clustering, CBLOF uses a heuristic to classify the resulting clusters into large and small clusters. Finally, an anomaly score is computed by the distance of each instance to its cluster center multiplied by the instances belonging to its cluster. For small clusters, the distance to the closest large cluster is used. The procedure of using the amount of cluster members as a scaling factor should estimate the local density of the clusters as stated by the authors. We showed in previous work that this assumption is not true [50] and might even result in a incorrect density estimation. Therefore, we additionally evaluate a modified version of CBLOF which simply neglects the weighting, referred to as unweighted-CBLOF (uCBLOF) in the following. The results of uCBLOF using a simple two-dimensional dataset are visualized in Fig 7, where the color corresponds to the clustering result of the preceding k-means clustering algorithm. Similar to the nearest-neighbor based algorithms, the number of initial clusters k is also a critical parameter. Here, we follow the same strategy as for the nearest-neighbor based algorithms and evaluate many different k values. Furthermore, k-means clustering is a non-deterministic algorithm and thus the resulting anomaly scores can be different on multiple runs. To this end we follow a common strategy, which is to apply k-means many times on the data and pick the most stable result. However, clustering-based anomaly detection algorithms are very sensitive to the parameter k, since adding just a single additional centroid might lead to a very different outcome.

# <베이스라인>

https://dacon.io/competitions/official/235930/codeshare/5236?page=1&dtype=recent

# <비슷한 케이스에 대한 케글 경진대회>

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?datasetId=310&searchQuery=unsu

# <관련해서 실험이나 테스트 해본 결과나 방법에 관해서는 아래 링크를 통해서 작성>

https://docs.google.com/document/d/1kMdX-Eaauec9xB_LiAfIercCNIQwABE1VncGrjV5m0Y/edit?usp=sharing

# <지도학습/비지도학습>

https://blog.naver.com/tommybee/222651270370
https://mangastorytelling.tistory.com/8471

- 지도학습

분류: 출력 변수가 범주형 데이터일 때 사용(ex. 예/아니요, 남성/여성, 참/거짓, 2개 이상의 클래스 포함)

회귀: 출력 변수가 실수 또는 연속 값일 때 사용. 둘 이상의 변수 사이의 관계를 가짐. (ex. 경력에 따른 급여, 키에 따른 체중, 한 변수의 변경이 다른 변수의 변경과 연관 됨.)

---
---
---
2주차 스터디 - 파라미터 조절 실험
===
---
[https://dacon.io/codeshare/5694?dtype=recent](https://dacon.io/codeshare/5694?dtype=recent)

EE = EllipticEnvelope(contamination=val_contamination, random_state=42)
에 support_fraction = 0.994 추가로 파라미터 조정
EE = EllipticEnvelope(support_fraction = 0.994, contamination=val_contamination, random_state=42)

<원래>
정확도: 0.9985, 정밀도: 0.6150, 재현율: 0.5996, F1:0.6068
<바꾼 후>
정확도: 0.9996, 정밀도: 0.9229, 재현율: 0.8666, F1:0.8928
향상된 것을 볼 수 있음..

대략 0.994 보다 작으면 F1 스코어가 떨어지면 0.994보다 크면 똑같음

<cf>
support_fraction  (float, default=None)
The proportion of points to be included in the support of the raw MCD estimate. If None, the minimum value of support_fraction will be used within the algorithm: [n_sample + n_features + 1] / 2. Range is (0, 1).
원시 MCD 추정치의 지원에 포함될 점의 비율입니다. 없음인 경우 support_fraction의 최소값은 알고리즘 내에서 사용됩니다. [n_sample + n_features + 1] / 2. 범위는 (0, 1)입니다.

IsolationForest
오차 행렬
[[28415    17]
[   18    12]]
정확도: 0.9988, 정밀도: 0.7066, 재현율: 0.6997, F1:0.7031

IsolationForest 150
오차 행렬
[[28413    19]
[   17    13]]
정확도: 0.9987, 정밀도: 0.7028, 재현율: 0.7163, F1:0.7094

---
---
---
3주차 스터디 - 모델 및 파라미터 실험
===
---
**IsolationForest**

오차 행렬

[[28420 12]

[ 19 11]]

정확도: 0.9989, 정밀도: 0.7388, 재현율: 0.6831, F1:0.7073

-------------------------------------------------

**LocalOutlierFactor**

오차 행렬

[[28401 31]

[ 30 0]]

정확도: 0.9979, 정밀도: 0.4995, 재현율: 0.4995, F1:0.4995

-------------------------------------------------

**KMeans**

오차 행렬

[[26368 2064]

[ 15 15]]

정확도: 0.9270, 정밀도: 0.5033, 재현율: 0.7137, F1:0.4881

-------------------------------------------------

**OneClassSVM**

오차 행렬

[[28168 264]

[ 16 14]]

정확도: 0.9902, 정밀도: 0.5249, 재현율: 0.7287, F1:0.5430

-------------------------------------------------

**EllipticEnvelope**

오차 행렬

[[28407 25]

[ 20 10]]

정확도: 0.9984, 정밀도: 0.6425, 재현율: 0.6662, F1:0.6535

-------------------------------------------------

**vote**

오차 행렬

[[28423 9]

[ 19 11]]

정확도: 0.9990, 정밀도: 0.7747, 재현율: 0.6832, F1:0.7198

-------------------------------------------------

**contamination, random_state : 안바꾼게 제일 남. 바꾸면 오히려 스코어 떨어짐.**

### **<KMeans>**

**random_state :** 안바꾼게 젤 남. 바꾸면 떨어짐

**n_init :** 4보다 크면 똑같고 작으면 떨어짐

**algorithm** : 차이 없음

**copy_x (default=True) :** 차이 없음

**max_iter (default=300) :** 차이 없음 (100, 500, 1000)

**verbose (default=0) :** 차이 없음 (1,2, 50,100, 300, 500)

**tol (default=1e-4) :** 별 차이 없어 보임 (0, 0.00001, 5)

**init{‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’ :** 바꾸면 더 떨어짐

### **<ElipticEnvelope>**

https://dacon.io/codeshare/5694?dtype=recent

**contamination :** 안바꾼게 젤 남. 바꾸면 떨어짐

**random_state :** 안바꾼게 젤 남. 바꾸면 떨어짐

**store_precision = False (default = True) :** 차이 없음

**assume_centered = True (default = False) :** 떨어짐;;;

**support_fraction = 0.994 (추가):**

<원래>

정확도: 0.9984, 정밀도: 0.6425, 재현율: 0.6662, F1:0.6535

<바꾼 후>

정확도: 0.9987, 정밀도: 0.6893, 재현율: 0.6830, F1:0.6861

향상된 것을 볼 수 있음

대략 0.994 보다 작으면 F1 스코어가 떨어지면 0.994보다 크면 똑같음

### **<IsolationForest>**

**contamination :** 안바꾼게 젤 남. 바꾸면 떨어짐

**random_state :** 안바꾼게 젤 남. 바꾸면 떨어짐

**verbose :** 차이 없음

**n_estimators :** 안바꾼게 젤 남. 바꾸면 떨어지거나 똑같음.

**bootstrap = True (default=False) :** 차이 없음

**warm_start = True (default=False) :** 차이 없음

**n_jobs (int, default=None) :** 차이 없음 (2, 50, 100, 125,300,1000)

**max_features = 0.5 or 0.4 :**

<원래>

정확도: 0.9989, 정밀도: 0.7388, 재현율: 0.6831, F1:0.7073

<바꾼 후>

정확도: 0.9989, 정밀도: 0.7497, 재현율: 0.6831, F1:0.7113

(0.1, 0.6, 0.9, 1.0) : 떨어짐

### **<LocalOutlierFactor>**

**algorithm :** 차이 없음

**leaf_size (default=30) :** 차이 없음 (2, 10,100,1000)

**p (default=2) :** 1 => 차이 없음 2보다 큰 수 => 너무 오래 걸려서 포기

**n_jobs (default=None) :** 차이 없음 (2, 50, 100)

**n_neighbors (default = 20) = 270 :**

<원래>

정확도: 0.9979, 정밀도: 0.4995, 재현율: 0.4995, F1:0.4995

<바꾼 후>

정확도: 0.9991, 정밀도: 0.7882, 재현율: 0.7498, F1:0.7676

### **<OneClassSVM>**

정확도: 0.9902, 정밀도: 0.5249, 재현율: 0.7287, F1:0.5430

**degree (default=3) :** 차이 없음 (10, 50, 100, 300, 500, 1000)

**shrinking (default=True) :** 차이 없음

**cache_size (float, default=200) :** 차이없음 (10, 100, 500, 1000) 차이없음

**verbose (default=False) :** 차이 없음

**kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} (default=’rbf’) :**

‘linear’ : 떨어짐

‘poly’ : 떨어짐

‘precomputed’ : 적용 안됨.

kernel = ‘sigmoid’

<원래> ("rbf")

정확도: 0.9902, 정밀도: 0.5249, 재현율: 0.7287, F1:0.5430

<바꾼 후>

정확도: 0.9983, 정밀도: 0.6407, 재현율: 0.6828, F1:0.6590

**coef0 (default=0.0) :** ‘sigmoid랑 ’poly' 일때만 유효

coef0 = 1.0

<'sigmoid‘ 일 때>

정확도: 0.9983, 정밀도: 0.6407, 재현율: 0.6828, F1:0.6590

<coef0까지 적용 후>

정확도: 0.9991, 정밀도: 0.7823, 재현율: 0.7165, F1:0.7450

0.8 = 0.7450

0.9 = 0.7405

1.5 = 0.6751

1.3 = 0.4995

2.0 = 0.6783

0.15 = 0.6771

1.05 = 0.4995

### **<cf>**

support_fraction (float, default=None)

The proportion of points to be included in the support of the raw MCD estimate. If None, the minimum value of support_fraction will be used within the algorithm: [n_sample + n_features + 1] / 2. Range is (0, 1).

원시 MCD 추정치의 지원에 포함될 점의 비율입니다. 없음인 경우 support_fraction의 최소값은 알고리즘 내에서 사용됩니다. [n_sample + n_features + 1] / 2. 범위는 (0, 1)입니다.
