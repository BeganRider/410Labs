import warnings
from multiprocessing.spawn import set_executable

from fontTools.misc.arrayTools import sectRect

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


df_titanic=pd.read_csv('Titanic.csv')
print(df_titanic)

print(df_titanic.info())

df_titanic.boxplot(column=['Age'])   #boxplot
plt.show()

sns.countplot(x="Sex",hue="Sex", data=df_titanic)  #countplot
plt.show()

sns.countplot(x="Pclass",hue="Pclass", data=df_titanic)
plt.show()

sns.countplot(x="Pclass", hue="Sex", data=df_titanic)
plt.show()

def titanic_children(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

df_titanic['person'] = df_titanic[['Age','Sex',]].apply(titanic_children, axis=1)
print(df_titanic.head(10))



sns.countplot(x="Pclass", hue="person", data=df_titanic)
plt.show()


as_fig = sns.FacetGrid(df_titanic,hue='Pclass',aspect=5)
as_fig.map(sns.kdeplot,'Age',shade=True)
oldest=df_titanic['Age'].max()
as_fig.set(xlim=(0,oldest))
as_fig.add_legend()
plt.show()


print(df_titanic.columns[df_titanic.isnull().any()])


print(df_titanic[df_titanic.columns[df_titanic.isnull().any()]].isnull().sum())

print(df_titanic['Age'].mean())

df_titanic['Age']= df_titanic['Age'].fillna(df_titanic['Age'].mean())
df_titanic['Fare']= df_titanic['Fare'].fillna(df_titanic['Fare'].mean())

df_titanic.drop('Cabin',axis=1,inplace=True)
sns.countplot(x="Embarked",hue="Pclass",data=df_titanic)
plt.show()

df_titanic['Embarked']=df_titanic['Embarked'].fillna('S')
print(df_titanic.isnull().values.any())

sns.countplot(x="Embarked",hue="Pclass",data=df_titanic)
plt.show()

sns.countplot(x="Survived", hue="Survived", data=df_titanic)
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=df_titanic)
plt.show()

sns.pointplot(x="Pclass", y="Survived", hue="person",data=df_titanic)
plt.show()

sns.set_theme(style="ticks")
sns.pairplot(df_titanic, hue='Survived')
plt.show()

df1_titanic=df_titanic[["Survived","Pclass","Age","SibSp","Parch","Fare"]]

corrmat = df1_titanic.corr()
print(corrmat)

sns.heatmap(corrmat,annot=True, linewidth=.5)
plt.show()

sns.lmplot(x="Age",y="Survived",hue="Pclass",data=df_titanic)
plt.show()

sns.lmplot(x="Age",y="Survived",hue="Embarked",data=df_titanic)
plt.show()



