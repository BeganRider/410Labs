import pandas as pd

#Load Titanic data set (Titanic.csv) into a Pandas DataFrame

df_titanic = pd.read_csv('Titanic.csv')

print(df_titanic)

#Display the first 5 rows of the dataset
print(df_titanic.head(n=5))

#Provide a summary of the dataset, including basic statistics

print(df_titanic.describe())

#Identify and display the data types of each column

print(df_titanic.info())

#Using a for loop, prints the values of two specific columns of the dataset side by side

print("Age \t Pclass")
for index, row in df_titanic.iterrows():
    print(row['Age'],"\t",row['Pclass'])
