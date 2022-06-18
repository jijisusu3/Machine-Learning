import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://docs.google.com/uc?export=download&id=1Fcwuh7HN0ii3DF7sK0TVl_oUwIh2wmIo',encoding='CP949' )

df.head()

df.columns

df.columns = ['id', '시간', '온도', '비', '풍속','습도', '가시성', '오존농도', '미세먼지', '초미세먼지', '이용자수']

df.head()

#시간대별 이용자수
plt.rc("font", family="Malgun Gothic")
df.groupby(["시간"], as_index = False)["이용자수"].mean().plot.bar()

# 피어슨 상관계수란 두 변수 사이의 직선관계의 강도를 나타낸다. 
# 이것은 x로부터 y를 에측할 수 있는 정도를 의미하며, 
# 일반적으로

# -1 < r < 0.7 : 강한 음의 선형관계
# -0.7 < r < -0.3 : 뚜렷한 음의 선형관계

# -0.3 < r < -0.1 : 약한 음의 선형관계
# -0.1 < r < 0.1 : 거의 무시될 수 있는 선형관계

# 0.1 < r < 0.3 : 약한 양의 상관관계
# 0.3 < r < 0.7 : 뚜렷한 양의 선형관계
# 0.7 < r < 1 : 강한 양의 선형관계 를 의미한다. 

# 온도와 이용자 수의 상관관계는 온도 1도 당 이용자수 0.619404 이므로 
# 뚜렷한 양의 선형관계를 나타냄을 알 수 있다.


plt.scatter(df['시간'], df['이용자수'])
df[["시간", '이용자수']].corr(method = 'pearson')

plt.scatter(df['온도'], df['이용자수'])
df[["온도", '이용자수']].corr(method = 'pearson')

df.groupby(["온도"])["이용자수"].mean().plot()

plt.scatter(df['습도'], df['이용자수'])
df[['습도', '이용자수']].corr(method = 'pearson')

plt.scatter(df['가시성'], df['이용자수'])
df[["가시성", '이용자수']].corr(method = 'pearson')

#강수 유무에 따른 이용량 분석
plt.scatter(df['비'], df['이용자수'])
df[['비', '이용자수']].corr(method = 'pearson')

#풍속에 따른 이용량 분석
plt.scatter(df['풍속'], df['이용자수'])
df[['풍속', '이용자수']].corr(method = 'pearson')

plt.scatter(df['오존농도'], df['이용자수'])
df[["오존농도", '이용자수']].corr(method = 'pearson')

#미세먼지에 따른 이용량 분석:
plt.scatter(df["미세먼지"], df["이용자수"])
df[["미세먼지", "이용자수"]].corr(method = "pearson")

#초미세먼지에 따른 이용량 분석
plt.scatter(df["초미세먼지"], df["이용자수"])
df[["초미세먼지", "이용자수"]].corr(method = "pearson")

df.drop(["id","비", "가시성","미세먼지", "초미세먼지"], axis = 1, inplace = True)

df=df.astype({'이용자수':'int'})
df

df = df.dropna()
df

from sklearn.model_selection import train_test_split # sklearn의 train_test_split을 사용하면 라인 한줄로 손쉽게 데이터를 나눌 수 있음
train, test = train_test_split(df, test_size=0.2)

train_x = np.array(train.iloc[:,:-1])
train_y = np.array(train.iloc[:, -1])

test_x = np.array(test.iloc[:, :-1])
test_y = np.array(test.iloc[:, -1])

# 과학적 표기법 대신 소수점 6자리까지 나타낸다.
np.set_printoptions(precision=6, suppress=True)

test_x
train_y
print(len(test_y))

#SVM 알고리즘을 통한 기계학습
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=100, gamma=0.001)

train_x = train[['시간', '온도', '풍속','습도', '오존농도']]
train_y = train['이용자수']
svm_model.fit(train_x, train_y)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

test_x = test[['시간', '온도', '풍속','습도', '오존농도']]
test_y = np.array(test.iloc[:, -1])
y_true, y_pred = test_y, svm_model.predict(test_x)


print(y_true)
print()
print(y_pred)

right_count = 0

for i in range(len(y_true)):
  if np.abs(y_true[i] - y_pred[i]) <= 20:
    right_count += 1

accuracy = right_count / len(y_true)

print(accuracy)

# KNN 알고리즘을 통한 기계학습
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

x_train = train[['시간','온도','풍속','습도','오존농도']] 
y_train = train[['이용자수']]  

print(x_train)
print(y_train.values)
print(y_train.values.ravel())

knn = KNeighborsClassifier(n_neighbors=2) 
knn.fit(x_train, y_train.values.ravel())
test_x = test[['시간', '온도', '풍속','습도', '오존농도']]
test_y = np.array(test.iloc[:, -1])
y_true, y_pred = test_y, knn.predict(test_x)

print(y_true)
print()
print(y_pred)

right_count = 0
for i in range(len(y_true)):
  if np.abs(y_true[i] - y_pred[i]) <= 20:
    right_count += 1

accuracy = right_count / len(y_true)

print(accuracy)

