# 고객 대출등급 분류 해커톤

https://dacon.io/competitions/official/236214/overview/description

- **task**: 고객의 대출등급을 예측하는 AI 알고리즘 개발
- **metric**: Macro F1
- **최종 결과**: Private 18위 (18/784)
---
---
---
# 문제해결과정

# 문제 정의

고객의 대출등급을 예측하는 AI 알고리즘 개발

<br/>

# EDA

- 참고
    - [https://dacon.io/competitions/official/236214/codeshare/9573?page=1&dtype=recent](https://dacon.io/competitions/official/236214/codeshare/9573?page=1&dtype=recent)
    - [https://dacon.io/competitions/official/236214/mysubmission](https://dacon.io/competitions/official/236214/mysubmission)
    - 결측치: 없음
    - 피처 별로 boxplot, countplot kdeplot 등 간단한 plot 시각화
    - 상관관계
        
        <img src="https://github.com/yebinchoi67/DACON_competition/blob/main/%EA%B3%A0%EA%B0%9D_%EB%8C%80%EC%B6%9C%EB%93%B1%EA%B8%89_%EB%B6%84%EB%A5%98_%ED%95%B4%EC%BB%A4%ED%86%A4/images/Untitled.png" width="70%" height="70%"/>

        
    - **대출금액, 총상환원금, 총상환이자, 연간소득의 상관성이 높음.**
        - 대출 금액이 클수록 갚을 원금이랑 이자는 당연히 높아질테니 비율로 피처 변환하기로 결정.
    

    <br/>
# 데이터 전처리

### 1. 이상치

<img src="https://github.com/yebinchoi67/DACON_competition/blob/main/%EA%B3%A0%EA%B0%9D_%EB%8C%80%EC%B6%9C%EB%93%B1%EA%B8%89_%EB%B6%84%EB%A5%98_%ED%95%B4%EC%BB%A4%ED%86%A4/images/Untitled%201.png" width="50%" height="50%"/>

- 부채_대비_소득_비율에서 1개 drop

    <br/>
### 2. 피처 엔지니어링

- **피처 생성**
    - **총상환원금비율, 총상환이자비율:** '대출금액'이 많을수록 '총상환원금'과 '총상환이자'는 당연히 수가 클거라 판단 후 '총상환원금', '총상환이자'를 '대출금액'으로 나누어서 비율로 보기로 결정
    - **연간소득/대출금액:** 대출금액에 따른 연간소득을 살펴보았을 때 본인의 소득 수준에서 대출을 얼마나 하는지 알 수 있을 것이라 판단 후 '연간소득'을 '대출금액'으로 나누어서 새 피처를 생성
    - **연체계좌수/총계좌수:** 총계좌 중 얼마나 많은 계좌가 연체되었는지의 정보를 통해 대출등급을 추정할 수 있을 것이라 판단 후 '연체계좌수'를 '총계좌수'로 나누어서 새 피처를 생성
    - **연간소득/총계좌수:** 소득에 따라 계좌를 얼마나 가지고 있는지의 정보를 통해 대출등급을 추정할 수 있을 것이라 판단 후 (돈은 없고 계좌만 많은 사람 등등) '연간소득'를 '총계좌수'로 나누어서 새 피처를 생성
    (*연간소득 스케일링 진행 후 피처 생성)*
- **drop:** 피처 생성 후 더이상 필요 없을 것 같은 피처 drop ('연체계좌수', '총계좌수', '총상환이자', '총상환원금')

    <br/>
### 3. 스케일링

- 연속형 데이터들 그래프로 살펴보니 분포가 너무 한쪽으로 치우친 피처들이 존재해서 스케일링 결정
(전체적으로 skewness가 큰 데이터셋)
- '연간소득', '부채_대비_소득_비율', '총상환원금비율', '총상환이자비율', '총연체금액’

<img src="https://github.com/yebinchoi67/DACON_competition/blob/main/%EA%B3%A0%EA%B0%9D_%EB%8C%80%EC%B6%9C%EB%93%B1%EA%B8%89_%EB%B6%84%EB%A5%98_%ED%95%B4%EC%BB%A4%ED%86%A4/images/Untitled%202.png" width="60%" height="60%"/>

- **<<고려 스케일러 목록>>**
    1. StandardScaler
    2. MinMaxScaler
    3. **log 변환**
    4. 제곱근 변환
    5. box-cox 변환 
    
    **➡️ log 변환 사용**
    
<img src="https://github.com/yebinchoi67/DACON_competition/blob/main/%EA%B3%A0%EA%B0%9D_%EB%8C%80%EC%B6%9C%EB%93%B1%EA%B8%89_%EB%B6%84%EB%A5%98_%ED%95%B4%EC%BB%A4%ED%86%A4/images/Untitled%203.png" width="40%" height="40%"/>

**log 변환**

<img src="https://github.com/yebinchoi67/DACON_competition/blob/main/%EA%B3%A0%EA%B0%9D_%EB%8C%80%EC%B6%9C%EB%93%B1%EA%B8%89_%EB%B6%84%EB%A5%98_%ED%95%B4%EC%BB%A4%ED%86%A4/images/Untitled%204.png" width="40%" height="40%"/>

**box-cox 변환**


<br/>
    
### 4. 피처 처리

1. **근로기간**
    - 숫자형으로 변환
    - Unknown: 평균값 대체
    - ‘< 1 year’, '<1 year’: 0으로 대체
    - 10+ : 10으로 대체
2. **주택소유상태**
    - ANY(1개 존재)를 MORTGAGE로 변환 (제일 많은 것인)
3. **대출 목적**
    - 부채통합(55150), 신용카드(24500), 기타(15837), 주택(807)의 4범주로 나눔.

    <br/>
### **5. 라벨 인코딩**

- 대출기간, 주택소유상태, 대출목적
- 대출등급(타겟)

    <br/>
# CleanLab

- dataset의 문제를 자동으로 감지해주는 오픈소스
- label issue (198개), near duplicate issue(194개) drop 해주고 학습 진행

    <br/>
# 학습 및 예측

- **K-fold를 이용한 voting 앙상블 모델**
    - 5-fold
    - xgboost
    - 각 fold에서 학습과 예측을 모두 진행 후 해당 결과들을 모두 모아 votting해서 최종 예측 생성
