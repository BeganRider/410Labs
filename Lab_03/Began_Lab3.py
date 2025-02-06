import warnings
from multiprocessing.spawn import set_executable

from fontTools.misc.arrayTools import sectRect

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

titanic_df=pd.read_csv("Titanic.csv")

titanic_df['Age']= titanic_df['Age'].fillna(titanic_df['Age'].mean())

#find and diplay mean median and standard deviation of age column


print('median : \t',titanic_df['Age'].median())
print('mean: \t', titanic_df['Age'].mean())
print('STD : \t', titanic_df['Age'].std())

print(titanic_df.head())

sns.set_theme(style="ticks")
sns.pairplot(titanic_df, hue='Age')
plt.show()

sns.countplot(x="Survived", hue="Age", data=titanic_df)
plt.show()

#drop the first column containining PassengerId

titanic_df.drop(columns=['PassengerId'], inplace=True)
print(titanic_df.head())

titanic_df.to_csv("TitanicModified.csv", index=False)