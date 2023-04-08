---
layout: single
title:  "titanic 머신러닝"
categories: jupyter
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>
# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:27.101083Z","iopub.execute_input":"2023-04-08T06:55:27.101406Z","iopub.status.idle":"2023-04-08T06:55:27.109564Z","shell.execute_reply.started":"2023-04-08T06:55:27.101375Z","shell.execute_reply":"2023-04-08T06:55:27.108403Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # **데이터 구하기**
# ---
# 어떤 사람들이 생존할 가능성이 높은지를 예측하는 모델훈련을 위한 데이터 다운로드 및 적재

# %% [markdown]
# ## **데이터 다운로드**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:27.111142Z","iopub.execute_input":"2023-04-08T06:55:27.111641Z","iopub.status.idle":"2023-04-08T06:55:27.167105Z","shell.execute_reply.started":"2023-04-08T06:55:27.111587Z","shell.execute_reply":"2023-04-08T06:55:27.166121Z"}}
# 데이터셋 불러오기
train = pd.read_csv('/kaggle/input/titanic/train.csv') # 훈련셋
test = pd.read_csv('/kaggle/input/titanic/test.csv') # 테스트셋

train.head()

# %% [markdown]
# * Survived: 생존 여부 / 0 = No, 1 = Yes
# * Pclass: 티켓 등급 / 1 = 1st, 2 = 2nd, 3 = 3rd
# * Sex: 성별
# * Age: 나이
# * Sibsp: 함께 탑승한 형제자매, 배우자의 수
# * Parch: 함께 탑승한 부모, 자식의 수
# * Ticket: 티켓 번호
# * Fare: 운임(in pounds)
# * Cabin: 객실 번호
# * Embarked: 탑승 항구 / C = Cherbourg, Q = Queenstown, S = Southampton

# %% [markdown]
# ## **데이터 구조 훑어보기**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:27.168590Z","iopub.execute_input":"2023-04-08T06:55:27.168937Z","iopub.status.idle":"2023-04-08T06:55:28.251667Z","shell.execute_reply.started":"2023-04-08T06:55:27.168890Z","shell.execute_reply":"2023-04-08T06:55:28.250638Z"}}
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts() # 생존자수
    dead = train[train['Survived'] == 0][feature].value_counts() # 사망자수

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    if(feature != 'Survived'):
        for i, index in enumerate(feature_index):
            plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
            plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
            plt.title(str(index) + '\'s ratio')
    
        plt.show()
        
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', figsize=(10, 5))

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:28.253987Z","iopub.execute_input":"2023-04-08T06:55:28.254346Z","iopub.status.idle":"2023-04-08T06:55:28.281408Z","shell.execute_reply.started":"2023-04-08T06:55:28.254301Z","shell.execute_reply":"2023-04-08T06:55:28.280342Z"}}
train.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:28.282558Z","iopub.execute_input":"2023-04-08T06:55:28.282806Z","iopub.status.idle":"2023-04-08T06:55:28.293316Z","shell.execute_reply.started":"2023-04-08T06:55:28.282776Z","shell.execute_reply":"2023-04-08T06:55:28.292264Z"}}
train.isnull().sum()

# %% [markdown]
# 우선, `Age`, `Cabin`, `Embarked` 속성에 결측치가 있음을 확인했다. 특히, `Cabin`은 결측치의 비율이 전체의 77%이므로 일단 무시하는것으로 한다. `Age`의 경우 중간값으로 결측치를 채우는 것으로 한다.
# 
# 또한 `Name`과 `Ticket` 속성은 모델이 사용할 수 있는 유용한 숫자로 변환하기 어렵기 때문에 이 특성 또한 무시한다.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:28.294934Z","iopub.execute_input":"2023-04-08T06:55:28.295275Z","iopub.status.idle":"2023-04-08T06:55:28.454419Z","shell.execute_reply.started":"2023-04-08T06:55:28.295238Z","shell.execute_reply":"2023-04-08T06:55:28.453230Z"}}
# 생존자 비율
pie_chart('Survived')

# %% [markdown]
# 위의 차트를 보면 전체 탑승객의 약 60% 정도가 사망했다는 것을 알 수 있다.

# %% [markdown]
# ### **범주형 특성**
# 
# * Sex
# * Pclass
# * Embarked

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:28.456026Z","iopub.execute_input":"2023-04-08T06:55:28.456632Z","iopub.status.idle":"2023-04-08T06:55:28.728028Z","shell.execute_reply.started":"2023-04-08T06:55:28.456576Z","shell.execute_reply":"2023-04-08T06:55:28.726880Z"}}
# 성별
pie_chart('Sex')

# %% [markdown]
# `male`이 `female`보다 타이타닉 호에 많이 탑승했으며, `male`보다 `female`의 생존 비율이 높다는 것을 알 수 있다.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:28.730084Z","iopub.execute_input":"2023-04-08T06:55:28.730467Z","iopub.status.idle":"2023-04-08T06:55:29.122732Z","shell.execute_reply.started":"2023-04-08T06:55:28.730425Z","shell.execute_reply":"2023-04-08T06:55:29.121462Z"}}
# 티켓 등급
pie_chart('Pclass')

# %% [markdown]
# 타이타닉 호에 탑승한 사람 중 `Pclass`가 3인사람, 즉 티켓 등급이 가장 낮은 사람들이이 55% 정도로 가장 많았으며, 티켓 등급이 1인 사람들이 그 다음으로 많았고 2인 사람들이 가장 적었다. 또한 티켓 등급이 높을수록(숫자가 작을수록) 생존 비율이 높다는 것을 알 수 있다.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:29.124862Z","iopub.execute_input":"2023-04-08T06:55:29.125509Z","iopub.status.idle":"2023-04-08T06:55:29.482396Z","shell.execute_reply.started":"2023-04-08T06:55:29.125456Z","shell.execute_reply":"2023-04-08T06:55:29.481124Z"}}
# 탑승 항구
pie_chart('Embarked')

# %% [markdown]
# Southampton, Cherbourg, Queenstown 순으로 탑승한 사람이 많았으며, Cherbourg을 제외한 다른 두 항구에서 탑승한 사람들은 사망자 비율이 생존자 비율보다 높았다.

# %% [markdown]
# ### **수치형 특성**
# 
# * SibSp
# * Parch

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:29.486458Z","iopub.execute_input":"2023-04-08T06:55:29.487222Z","iopub.status.idle":"2023-04-08T06:55:29.811023Z","shell.execute_reply.started":"2023-04-08T06:55:29.487165Z","shell.execute_reply":"2023-04-08T06:55:29.809989Z"}}
# 함께 탑승한 형제자매, 배우자의 수
bar_chart('SibSp')

# %% [markdown]
# 생존자 중에서는 혼자 탑승한 사람의 수가 가장 많다. 하지만, 혼자 탑승한 사람의 사망자 대비 생존자의 비율은 매우 낮은 것을 알 수 있다.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:29.812342Z","iopub.execute_input":"2023-04-08T06:55:29.812656Z","iopub.status.idle":"2023-04-08T06:55:30.092200Z","shell.execute_reply.started":"2023-04-08T06:55:29.812620Z","shell.execute_reply":"2023-04-08T06:55:30.091070Z"}}
# 함께 탑승한 부모, 자식의 수
bar_chart('Parch')

# %% [markdown]
# `Parch` 특성은 `Sibsp`와 비슷한 비율을 보인다.

# %% [markdown]
# # **데이터 전처리**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.093860Z","iopub.execute_input":"2023-04-08T06:55:30.094219Z","iopub.status.idle":"2023-04-08T06:55:30.582429Z","shell.execute_reply.started":"2023-04-08T06:55:30.094171Z","shell.execute_reply":"2023-04-08T06:55:30.581461Z"}}
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# 수치형 데이터 파이프라인
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # 결측치 중간값으로 채움
    ("scaler", StandardScaler()) # 표준화
])

# 범주형 데이터 파이프라인
cat_pipeline = Pipeline([
    ("ordinal_encoder", OrdinalEncoder()),
    ("imputer", SimpleImputer(strategy="most_frequent")), # 결측치 최빈값으로 채움
    ("cat_encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))
])

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.583770Z","iopub.execute_input":"2023-04-08T06:55:30.584028Z","iopub.status.idle":"2023-04-08T06:55:30.595214Z","shell.execute_reply.started":"2023-04-08T06:55:30.583999Z","shell.execute_reply":"2023-04-08T06:55:30.594183Z"}}
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"] # 수치형 특성
cat_attribs = ["Pclass", "Sex", "Embarked"] # 범주형 특성

# 특성별 파이프라인 지정
preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.596775Z","iopub.execute_input":"2023-04-08T06:55:30.597030Z","iopub.status.idle":"2023-04-08T06:55:30.618754Z","shell.execute_reply.started":"2023-04-08T06:55:30.597000Z","shell.execute_reply":"2023-04-08T06:55:30.618099Z"}}
X_train = preprocess_pipeline.fit_transform(train)
X_train

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.619913Z","iopub.execute_input":"2023-04-08T06:55:30.620518Z","iopub.status.idle":"2023-04-08T06:55:30.625308Z","shell.execute_reply.started":"2023-04-08T06:55:30.620485Z","shell.execute_reply":"2023-04-08T06:55:30.624394Z"}}
y_train = train["Survived"] # label

# %% [markdown]
# # **모델 선택과 훈련**

# %% [markdown]
# ## **랜덤 포레스트 분류**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.626643Z","iopub.execute_input":"2023-04-08T06:55:30.626977Z","iopub.status.idle":"2023-04-08T06:55:30.929321Z","shell.execute_reply.started":"2023-04-08T06:55:30.626933Z","shell.execute_reply":"2023-04-08T06:55:30.928609Z"}}
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.930389Z","iopub.execute_input":"2023-04-08T06:55:30.931129Z","iopub.status.idle":"2023-04-08T06:55:30.965690Z","shell.execute_reply.started":"2023-04-08T06:55:30.931086Z","shell.execute_reply":"2023-04-08T06:55:30.964301Z"}}
# 테스트셋에 대하여 예측하기
X_test = preprocess_pipeline.transform(test)
y_pred = forest_clf.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:30.966999Z","iopub.execute_input":"2023-04-08T06:55:30.967303Z","iopub.status.idle":"2023-04-08T06:55:33.019711Z","shell.execute_reply.started":"2023-04-08T06:55:30.967267Z","shell.execute_reply":"2023-04-08T06:55:33.018797Z"}}
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

# %% [markdown]
# ## **SVC 모델**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.021339Z","iopub.execute_input":"2023-04-08T06:55:33.021888Z","iopub.status.idle":"2023-04-08T06:55:33.286513Z","shell.execute_reply.started":"2023-04-08T06:55:33.021826Z","shell.execute_reply":"2023-04-08T06:55:33.285570Z"}}
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()

# %% [markdown]
# SVC 모델이 랜덤포레스트로 예측하는 것보다 더 성능이 좋다.

# %% [markdown]
# 그러나 10겹 교차 검증의 평균 정확도만 보는 것이 아니라, 각 모델에 대한 10개의 점수를 모두 표시하고, 하위 사분위와 상위 사분위를 강조하는 상자 그림과 점수의 범위를 보여주는 "수염"을 표시해보자. `boxplot()` 함수는 이상치(fliers)를 탐지하고 수염에 포함하지 않는다. 특히, 하위 사분위가 $Q_1$이고 상위 사분위가 $Q_3$이면 사분위간 범위(상자높이) $IQR=Q_3-Q_1$ 이고 $Q_1-1.5*IQR$보다 낮은 점수는 이상치이며 $Q_3+1.5*IQR$보다 큰 점수도 이상치이다.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.287871Z","iopub.execute_input":"2023-04-08T06:55:33.288727Z","iopub.status.idle":"2023-04-08T06:55:33.474186Z","shell.execute_reply.started":"2023-04-08T06:55:33.288685Z","shell.execute_reply":"2023-04-08T06:55:33.473145Z"}}
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()

# %% [markdown]
# 랜덤 포레스트 분류기는 10겹중 하나에서 매우 높은 점수를 받았지만, 전반적으로 평균 점수가 낮을 뿐 아니라 격차도 커 SVM 분류기가 일반화될 가능성이 더 높아보인다.

# %% [markdown]
# # **성능 향상**

# %% [markdown]
# 이 결과를 개선하기 위해서 다음과 같이 할 수 있다.
# 
# * 교차 검증 및 그리드 탐색을 사용하여 더 많은 모델을 비교하고 하이퍼 파라미터 조정
# * 다음과 같이 특성 엔지니어링을 추가로 실행
#     * 수치 속성을 범주형 속성으로 변환
#         * ex) 연령 그룹마다 생존율이 매우 다르므로 연령 버킷 범주를 만들고 나이 대신 사용
#     * `SibSp`와 `Parch`를 그들의 합으로 대체
#     * `Survived` 속성과 잘 상관되는 이름 부분 식별
#     * `Cabin`열 사용
#         * ex) 첫 번째 문자를 이용하여 범주형 속성으로 처리

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.475499Z","iopub.execute_input":"2023-04-08T06:55:33.475874Z","iopub.status.idle":"2023-04-08T06:55:33.493664Z","shell.execute_reply.started":"2023-04-08T06:55:33.475832Z","shell.execute_reply":"2023-04-08T06:55:33.492655Z"}}
train2 = pd.read_csv('/kaggle/input/titanic/train.csv') # 훈련셋
test2 = pd.read_csv('/kaggle/input/titanic/test.csv') # 테스트셋

# %% [markdown]
# **수치 속성을 범주형 속성으로 변환**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.495245Z","iopub.execute_input":"2023-04-08T06:55:33.495798Z","iopub.status.idle":"2023-04-08T06:55:33.516157Z","shell.execute_reply.started":"2023-04-08T06:55:33.495756Z","shell.execute_reply":"2023-04-08T06:55:33.515079Z"}}
train2["AgeBucket"] = train2["Age"] // 15 * 15    # 15 단위로 연령 그룹 생성
train2[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()    # 그룸별 데이터 집계

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.517672Z","iopub.execute_input":"2023-04-08T06:55:33.517959Z","iopub.status.idle":"2023-04-08T06:55:33.523879Z","shell.execute_reply.started":"2023-04-08T06:55:33.517926Z","shell.execute_reply":"2023-04-08T06:55:33.523197Z"}}
test2["AgeBucket"] = test2["Age"] // 15 * 15    # 15 단위로 연령 그룹 생성

# %% [markdown]
# ---

# %% [markdown]
# **`SibSp`와 `Parch`를 그들의 합으로 대체**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.526782Z","iopub.execute_input":"2023-04-08T06:55:33.527079Z","iopub.status.idle":"2023-04-08T06:55:33.545019Z","shell.execute_reply.started":"2023-04-08T06:55:33.527023Z","shell.execute_reply":"2023-04-08T06:55:33.544001Z"}}
train2["Family"] = train2["SibSp"] + train2["Parch"]
train2[["Family", "Survived"]].groupby(['Family']).mean()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.546474Z","iopub.execute_input":"2023-04-08T06:55:33.546925Z","iopub.status.idle":"2023-04-08T06:55:33.557931Z","shell.execute_reply.started":"2023-04-08T06:55:33.546887Z","shell.execute_reply":"2023-04-08T06:55:33.557065Z"}}
test2["Family"] = test2["SibSp"] + test2["Parch"]

# %% [markdown]
# ---

# %% [markdown]
# **Survived 속성과 잘 상관되는 이름 부분 식별**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.559286Z","iopub.execute_input":"2023-04-08T06:55:33.559761Z","iopub.status.idle":"2023-04-08T06:55:33.594651Z","shell.execute_reply.started":"2023-04-08T06:55:33.559716Z","shell.execute_reply":"2023-04-08T06:55:33.593827Z"}}
train2['Title'] = train2.Name.str.extract(' ([A-Za-z]+)\.')    # 공백으로 시작하고 . 으로 끝나느 문자열 추출
    
pd.crosstab(train2['Title'], train2['Sex'])

# %% [markdown]
# 여기서 흔하지 않은 Title은 Other로 대체하고 중복되는 표현을 통일하자.

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.596127Z","iopub.execute_input":"2023-04-08T06:55:33.596454Z","iopub.status.idle":"2023-04-08T06:55:33.618666Z","shell.execute_reply.started":"2023-04-08T06:55:33.596410Z","shell.execute_reply":"2023-04-08T06:55:33.617775Z"}}
train2['Title'] = train2['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                           'Dona', 'Dr', 'Jonkheer','Lady',
                                           'Major', 'Rev', 'Sir'], 'Other')
train2['Title'] = train2['Title'].replace('Mlle', 'Miss')
train2['Title'] = train2['Title'].replace('Mme', 'Mrs')
train2['Title'] = train2['Title'].replace('Ms', 'Miss')

train2[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.623262Z","iopub.execute_input":"2023-04-08T06:55:33.623553Z","iopub.status.idle":"2023-04-08T06:55:33.629404Z","shell.execute_reply.started":"2023-04-08T06:55:33.623519Z","shell.execute_reply":"2023-04-08T06:55:33.628297Z"}}
# 추출한 Title 데이터를 학습하기 알맞게 String으로 변형
train2['Title'] = train2['Title'].astype(str)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.630792Z","iopub.execute_input":"2023-04-08T06:55:33.631472Z","iopub.status.idle":"2023-04-08T06:55:33.647685Z","shell.execute_reply.started":"2023-04-08T06:55:33.631422Z","shell.execute_reply":"2023-04-08T06:55:33.646896Z"}}
test2['Title'] = test2.Name.str.extract(' ([A-Za-z]+)\.') 

test2['Title'] = test2['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                           'Dona', 'Dr', 'Jonkheer','Lady',
                                           'Major', 'Rev', 'Sir'], 'Other')
test2['Title'] = test2['Title'].replace('Mlle', 'Miss')
test2['Title'] = test2['Title'].replace('Mme', 'Mrs')
test2['Title'] = test2['Title'].replace('Ms', 'Miss')

test2['Title'] = test2['Title'].astype(str)

# %% [markdown]
# ---

# %% [markdown]
# **전처리**

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.649073Z","iopub.execute_input":"2023-04-08T06:55:33.649826Z","iopub.status.idle":"2023-04-08T06:55:33.659873Z","shell.execute_reply.started":"2023-04-08T06:55:33.649776Z","shell.execute_reply":"2023-04-08T06:55:33.658733Z"}}
num_attribs2 = ["Family", "Fare"] # 수치형 특성
cat_attribs2 = ["AgeBucket", "Pclass", "Sex", "Embarked", "Title"] # 범주형 특성

# 특성별 파이프라인 지정
preprocess_pipeline2 = ColumnTransformer([
    ("num", num_pipeline, num_attribs2),
    ("cat", cat_pipeline, cat_attribs2),
])

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.662196Z","iopub.execute_input":"2023-04-08T06:55:33.662662Z","iopub.status.idle":"2023-04-08T06:55:33.692448Z","shell.execute_reply.started":"2023-04-08T06:55:33.662610Z","shell.execute_reply":"2023-04-08T06:55:33.691767Z"}}
X_train2 = preprocess_pipeline2.fit_transform(train2)
X_train2

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.695107Z","iopub.execute_input":"2023-04-08T06:55:33.695764Z","iopub.status.idle":"2023-04-08T06:55:33.700110Z","shell.execute_reply.started":"2023-04-08T06:55:33.695724Z","shell.execute_reply":"2023-04-08T06:55:33.699310Z"}}
y_train2 = train2["Survived"] # label

# %% [markdown]
# ---

# %% [markdown]
# **모델 훈련**

# %% [markdown]
# * 랜덤 포레스트

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.701298Z","iopub.execute_input":"2023-04-08T06:55:33.702059Z","iopub.status.idle":"2023-04-08T06:55:33.720139Z","shell.execute_reply.started":"2023-04-08T06:55:33.702004Z","shell.execute_reply":"2023-04-08T06:55:33.719154Z"}}
X_test2 = preprocess_pipeline2.transform(test2)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:33.721431Z","iopub.execute_input":"2023-04-08T06:55:33.722439Z","iopub.status.idle":"2023-04-08T06:55:35.743637Z","shell.execute_reply.started":"2023-04-08T06:55:33.722397Z","shell.execute_reply":"2023-04-08T06:55:35.742613Z"}}
forest_scores2 = cross_val_score(forest_clf, X_train2, y_train2, cv=10)
forest_scores2.mean()

# %% [markdown]
# * SVC

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:35.744853Z","iopub.execute_input":"2023-04-08T06:55:35.745104Z","iopub.status.idle":"2023-04-08T06:55:35.778130Z","shell.execute_reply.started":"2023-04-08T06:55:35.745075Z","shell.execute_reply":"2023-04-08T06:55:35.777088Z"}}
svm_clf.fit(X_train2, y_train2)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:35.779417Z","iopub.execute_input":"2023-04-08T06:55:35.779655Z","iopub.status.idle":"2023-04-08T06:55:35.795202Z","shell.execute_reply.started":"2023-04-08T06:55:35.779626Z","shell.execute_reply":"2023-04-08T06:55:35.794424Z"}}
svm_pred = svm_clf.predict(X_test2)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:35.796510Z","iopub.execute_input":"2023-04-08T06:55:35.796872Z","iopub.status.idle":"2023-04-08T06:55:36.046205Z","shell.execute_reply.started":"2023-04-08T06:55:35.796832Z","shell.execute_reply":"2023-04-08T06:55:36.045546Z"}}
svm_scores2 = cross_val_score(svm_clf, X_train2, y_train2, cv=10)
svm_scores2.mean()

# %% [markdown]
# 변형한 데이터셋에 대한 SVC 모델의 훈련 성능이 증가한 것을 알 수 있다.

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-04-08T06:55:36.047350Z","iopub.execute_input":"2023-04-08T06:55:36.047707Z","iopub.status.idle":"2023-04-08T06:55:36.056183Z","shell.execute_reply.started":"2023-04-08T06:55:36.047676Z","shell.execute_reply":"2023-04-08T06:55:36.055332Z"}}
submission = pd.DataFrame({
     "PassengerId": test2["PassengerId"],
     "Survived": svm_pred
})

submission.to_csv('submission_svc.csv', index=False)
