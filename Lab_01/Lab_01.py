import pandas as pd

#read xlsx

data_xlsx=pd.read_excel('Bank.xlsx' , index_col=0)
print(data_xlsx)

data_xlsx=pd.read_excel('Bank.xlsx' ,header = None)
print(data_xlsx)

data_xlsx=pd.read_excel('Bank.xlsx' , dtype={
    "EducLev": str,
    "JobGrade": float})
print(data_xlsx)

#read csv

data_csv=pd.read_csv('Bank.csv')
print(data_csv.head())

print(data_csv.describe())


data_csv.to_csv('Bank.csv')
print(data_csv.info())
data_csv['Gender'] = data_csv['Gender'].astype('category')
data_csv['PCJob']=data_csv['PCJob'].astype('category')
data_csv['Mgmt']=data_csv['Mgmt'].astype('category')
print(data_csv.info())
print(data_csv.describe(include='category'))


data_xlsx=pd.read_excel('Bank.xlsx')
print(data_xlsx.info())
for col in ['Gender','PCJob']:
    data_xlsx[col] = data_xlsx[col].astype('category')
print(data_xlsx.info())
