import pandas as pd

df = pd.read_csv('diabetes.csv')

print(df.describe())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

before = df['Pregnancies']
after_1 = scaler.fit_transform(df[['Pregnancies']])


# print(before)
# print(after)
# for b, a in zip(before, after_1):
#     print(f"before: {b}, after: {a}")


from sklearn.preprocessing import StandardScaler, RobustScaler 

scaler = StandardScaler()

after_2 = scaler.fit_transform(df[['Pregnancies']])

after_3 = RobustScaler().fit_transform(df[['Pregnancies']])

for b, a1, a2, a3 in zip(before, after_1, after_2, after_3):
    print(f"before: {b}, after_1: {a1}, after_2: {a2}, after_3: {a3}")
    
from sklearn.preprocessing import OrdinalEncoder  
data = pd.DataFrame(['XL', 'M', 'L', 'XL', 'XXL', 'XXXL', 'XXL', 'L'])
values = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']
encoder = OrdinalEncoder(categories=[values]) 
data_encoded = encoder.fit_transform(data)
print(data_encoded)
from matplotlib import pyplot as plt
import seaborn as sns

# plt.hist(df)
# df.hist()
print("================================================")
print(df.hist())
df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# plt.show()
print("================================================")

data_corr = df.corr()
print(data_corr)

# Display correlation matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

print("================================================")