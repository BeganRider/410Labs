import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)

df = pd.read_csv("data.csv")
print(df.head(5))

print(df.tail(5))

print(df.dtypes)

df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)

print(df.head(5))

df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })

print(df.head(5))

print(df.shape)
duplicate_rows_df=df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)

print(df.count())

df = df.drop_duplicates()
print(df.head(5))
print(df.count())

print(df.isnull().sum())

df=df.dropna()
print(df.count())

print(df.isnull().sum())

sns.boxplot(x=df['Price'])
plt.show()

sns.boxplot(x=df['HP'])
plt.show()

sns.boxplot(x=df['Cylinders'])
plt.show()

def remove_outlier(df, column):
    Q1= df[column].quantile(0.25)
    Q3= df[column].quantile(0.75)
    IQR=Q3-Q1

    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    df_no_outliers = df[(df[column]>=lower_bound) & (df[column]<=upper_bound)]
    return df_no_outliers

columns_to_check = ['Price', 'Cylinders', 'HP']

for column in columns_to_check:
    df=remove_outlier(df, column)

sns.boxplot(x=df['HP'])
plt.show()

df.Make.value_counts().nlargest(30).plot(kind='bar',figsize=(9,4))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make')
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()


df1= df[["Year","HP", "Cylinders", "MPG-H", "MPG-C", "Price"]]
plt.figure(figsize=(20,10))
c= df1.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
plt.show()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Make"]=le.fit_transform(df["Make"])

print(df.dtypes)

X = df[['HP','Cylinders','MPG-H', 'MPG-C', 'Price','Year']].values
Y = df["Make"].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {y_train}".format(y_train = len(Y_train)))
print("Length of y_test is: {y_test}".format(y_test = len(Y_test)))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
model=GradientBoostingClassifier()

model.fit(X_train,Y_train)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
score=accuracy_score(Y_test,pred)
print(f'Accuracy GBC: {round(score*100,2)}%')

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,Y_train)
pred = model.predict(X_test)
score2=accuracy_score(Y_test,pred)
print(f'Accuracy GNB: {round(score2*100,2)}%')

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)
pred = model.predict(X_test)
score=accuracy_score(Y_test,pred)
print(f'Accuracy RF: {round(score*100,2)}%')

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,Y_train)
pred = model.predict(X_test)
score=accuracy_score(Y_test,pred)
print(f'Accuracy SVM: {round(score*100,2)}%')

