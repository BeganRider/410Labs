import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import isnull

df=pd.read_csv('diabetes.csv')
print(df.shape)
print(df.head())
print(df.isnull)
print(df.isna)
#drop missing data
df=df.fillna(df.mean())
print(df.head())
print(df.shape)
#find mean, median, and standard deviation of age column

print('Age mean: ',df['Age'].mean())
print('Age median: ',df['Age'].median())
print('Age standard deviation: ',df['Age'].std())

correlation_matrix = df.corr()
# Create the heatmap
sns.heatmap(correlation_matrix,annot=True, linewidth=.5)
plt.show()

df_dropped=df.drop('Age',axis=1)
corrmap=df_dropped.corr()
sns.heatmap(corrmap,annot=True, linewidth=.5)
plt.show()

plt.hist(data=df, x="BMI", bins=90)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

sns.set_theme(style="ticks")
sns.pairplot(df, hue='Outcome')
plt.show()

df_dropped.to_csv('diabetesModified.csv')

