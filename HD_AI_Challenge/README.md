HD현대 AI Challenge
===
https://dacon.io/competitions/official/236158/overview/description
- **task:** 항만 內 선박 대기 시간 예측을 위한 선박 항차 데이터 분석 AI 알고리즘 개발
- **matric:** MAE (Mean Absolute Error, 평균 절대 오차)
===
===
===
# 문제해결 과정

# 문제 정의

**선박의 대기 시간 예측**

- **+) 접안**(배를 육지에 대는 것;Berthing) 전에 **해상에 정박** (해상에 닻을 바다 밑바닥에 내려놓고 운항을 멈추는 것;Anchorage)하는 시간을 대기시간으로 정의

# EDA

### 데이터 기본 분석

1. **데이터 총 개수:** 391939개
2. **피처 총 개수:** 22개(target 포함)
3. **피처 정리:**
    
    
    | ARI_CO | 도착항의 소속국가 |
    | --- | --- |
    | ARI_PO | 도착항의 항구명 |
    | SHIP_TYPE_CATEGORY | 선박 종류 |
    | DIST | 정박지와 접안지 사이의 거리 |
    | ATA | 실제 정박 시간(Actual Time of Arrival)UTC 사용. (국제 표준시) |
    | ID | 선박 ID |
    | BREADTH | 선박의 폭 |
    | BUILT | 선박의 연령 |
    | DEADWEIGHT | 선박의 재화중량톤수 |
    | DEPTH | 선박의 깊이 |
    | DRAUGHT | 흘수 높이(물속에 잠긴 선박의 깊이) |
    | GT | 용적톤수 값 |
    | LENGTH | 선박의 길이 |
    | SHIPMANAGER | 선박 소유주 |
    | FLAG | 선박의 국적 |
    | U_WIND | 풍향 u벡터 |
    | V_WIND | 풍향 v벡터 |
    | AIR_TEMPERATURE | 기온 |
    | BN | 보퍼트 풍력 계급(풍속) |
    | ATA_LT | 현지 정박시간(Local Time of Arrival) |
    | PORT_SIZE | 접안지 폴리곤 영역의 크기 |
    | CI_HOUR | 대기시간 (target 피처) |
4. **데이터 타입(object형):** ARI_CO, ARI_PO, ATA, ID, SHIP_TYPE_CATEGORY, SHIPMANAGER, FLAG
5. **None값:**
    - **BREADTH : 1개** (356484번 데이터)
    - **DEPTH : 1개** (356484번 데이터)
    - **DRAUGHT : 1개** (356484번 데이터)
    - **LENGTH: 1개** (356484번 데이터)
    - **U_WIND : 163688개**
    - **V_WIND : 163688개**
    - **AIR_TEMPERATURE : 164630개**
    - **BN : 163688개**
    
    **=> 전체적으로 기상 관련된 자료가 결측치가 굉장히 많음. (전체 데이터의 절반정도가 결측치)**
    

### 데이터 분포

![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled.png)

![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%201.png)

### 데이터 전처리 (임시)

1. **ATA**
    - year, month, hour, weekday로 분리
    - 작업 후 ATA는 제거
2. **카테고리형 컬럼 인코딩**
    - Label Encoder 적용
    - ARI_CO, ARI_PO, SHIP_TYPE_CATEGORY, ID, SHIPMANAGER, FLAG 에 적용
3. **결측치 처리**
    - drop: BREADTH, DEPTH 등 선박의 정보가 없는 356484번째 데이터 하나 삭제
    - 나머지 결측치는 mean으로 보간

### 모델 학습 및 특성 중요도 확인

1. **LGBM**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%202.png)
    
2. **Xgboost**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%203.png)
    
3. **RandomForest**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%204.png)
    
4. **CatBoost**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%205.png)
    

### 피처 제거, 결측치 보간 방법 실험

- Xgboost 사용.
- matric: MAE
- 5-fold
1. **기본적인 전처리(ATA 컬럼처리+카테고리형 라벨인코딩+선박외형 관련 결측치 존재하는 행 제거 등등)**
    - Validation : MAE: 56.329948442704506
2. **기본 전처리 + 결측치 처리 mean으로**
    - Validation : MAE: 56.287011066103005
3. **기본 전처리 +컬럼 삭제1**
    - 삭제 컬럼: ('ID', 'DEPTH', 'DRAUGHT', 'minute')
    - Validation : MAE: 56.21984024281208
4. **기본 전처리 +컬럼 삭제2 (결측치 많은 컬럼 다 삭제)**
    - 삭제 컬럼: ('ID', 'DEPTH', 'DRAUGHT', 'minute', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN’)
    - Validation : MAE: 56.028864216760304
5. **기본 전처리 + 결측치 knn보간**
    - Validation : MAE: 56.73492415099061
6. **기본 전처리 + 결측치 mice보간**
    - Validation : MAE: 56.38276786156872
7. **기본 전처리 + 판다스 선형보간**
    - Validation : MAE: 56.60900384813006
8. **기본 전처리(ATA 컬럼처리+카테고리형 라벨인코딩) + 기상컬럼 다 삭제**
    - 삭제 컬럼: ('U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN’)
    - Validation : MAE: 56.24177153099379
9. **기본+컬럼 삭제... +BREADTH** 
    - 삭제 컬럼: ('ID', 'DEPTH', 'DRAUGHT', 'minute', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'BREADTH’)
    - Validation : MAE: 55.9284166210835
10. **기본+컬럼 삭제... +BUILT** 
    - 삭제 컬럼: ('ID', 'DEPTH', 'DRAUGHT', 'minute', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'BREADTH', 'BUILT’)
    - Validation : MAE: 56.01877759496064
11. **기본+컬럼 삭제... +SHIPMANAGER** 
    - 삭제 컬럼: ('ID', 'DEPTH', 'DRAUGHT', 'minute', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'BREADTH', 'BUILT', 'SHIPMANAGER’)
    - Validation : MAE: 55.88073928358823

### 피처 별 그래프

1. **수치형 (DIST, DEADWEIGHT, GT, LENGTH, PORT_SIZE, BUILTB)**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%206.png)
    
2. **범주형 (ARI_CO, ARI_PO, SHIP_TYPE_CATEGORY, FLAG, SHIPMANAGER)**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%207.png)
    
3. **시간 데이터 (ATA_LT, year, month, day, hour, miniute, weekday)**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%208.png)
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%209.png)
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%2010.png)
    
4. **상관계수 히트맵**
    
    ![Untitled](https://github.com/yebinchoi67/DACON_competition/blob/ea5924c1bb5e768564e4d7c5b571257eba8971d5/HD_AI_Challenge/images/Untitled%2011.png)
    

# 최종 문제 해결 방법

### 데이터 전처리

1. **시간 데이터 처리**
    - year, month, hour, weekday로 분리
    - TIME_DIFFERENCE라는 피처로 ATA(UTC시간), ATA_LT(로컬 시간)의 시차에 관한 피처 생성
    - 작업 후 ATA는 제거
2. **카테고리형 컬럼 인코딩**
    - Label Encoder 적용
    - ARI_CO, ARI_PO, SHIP_TYPE_CATEGORY, ID, SHIPMANAGER, FLAG 에 적용
3. **결측치 처리**
    - drop: BREADTH, DEPTH 등 선박의 정보가 없는 356484번째 데이터 하나 삭제
4. **스케일링**
    - min / max의 범위가 너무 넓은 피처 스케일링 적용
    - MinMaxScaler 사용 (0~400) 범위로
    - GT, DEADWEIGHT 에 적용
5. **피처 생성**
    1. **BUILT_old :** BUILT(선박의 연령)을 이용해 노선을 나타내는 피처 생성
        - 노선인지 아닌지 이진 피처로 생성
        - 노선: 25년 초과면 1 아니면 0
        - 선박 평균 수명은 25~30년 이라고 함
    2. **DIST_CATE:** DIST(정박지와 접안지 사이의 거리)로 생성
        - DIST에 0인 값이 존재
        - 0 유무에 따라 이진 피처로 생성
    3. **SHIPMANAGER_RICH:** SHIPMANAGER로 생성(선박 소유주)
        - 1000개 넘는 선박 소유주 존재
        - 1000개 넘는 선박 소유 여부에 따라 이진 피처로 생성
    4. **bn_cate:** BN으로 생성
        - BN을 3개의 범주로 나눔
        - BN(보퍼트 풍력 계급)이 3 미만인 경우 0
        - BN(보퍼트 풍력 계급)이 3 이상이고 6 이하인 경 1
        - BN(보퍼트 풍력 계급)이 6 초과인 경우 2
    5. **hot:** AIR_TEMPERATURE로 생성
        - 폭염의 여부를 이진 피처로 생성
        - 폭염: 섭씨 35도 이상
    6. **cold:** AIR_TEMPERATURE로 생성
        - 한파의 여부를 이진 피처로 생성
        - 한파: 섭씨 -15도 이하
    7. **weekend :** weekday로 생성
        - 주말의 여부를 이진 피처로 생성
6. **Drop**
    - **Drop 목록:** ID, DEPTH, DRAUGHT, minute, U_WIND, V_WIND, AIR_TEMPERATURE, BN, BREADTH, BUILT, SHIPMANAGER, FLAG

### 모델 선정 및 최적화

- AutoML (AutoGluon) 사용
- [https://auto.gluon.ai/stable/index.html](https://auto.gluon.ai/stable/index.html)
- presets = medium_quality

### 최종 Score

- public: 47.89784
- private: 47.95251


