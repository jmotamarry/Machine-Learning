import pandas as pd
import matplotlib.pyplot as plt


pd.options.display.max_columns = 6

df = pd.read_csv('C:/Users/Jay Motamarry/Downloads/titanic.csv')
"""
print(df.head())
print(df.describe())
print(df['Fare'])
print(df[['Age', 'Sex', 'Survived']].head())
df['Male'] = df['Sex'] == 'male'
print(df[['Sex', 'Male']])
print(df['Fare'].values)    # converts to a numpy array
print(df[['Fare', 'Age', 'Pclass']].values)    # converts to a numpy array that is 2D
arr = df[['Fare', 'Age', 'Pclass']].values  # makes a 2D np array in arr
print(arr.shape)  # prints (row, col)
"""
plt.scatter(df['Age'], df['Fare'])  # plot age on x and fare on y
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])  # plot age on x, fare on y, and color based on Pclass
plt.xlabel('Age')   # label x as 'Age'
plt.ylabel('Fare')  # label y as 'Fare'
plt.plot([0, 80], [85, 5])  # plots a line from (0, 85) to (80, 5)

plt.show()  # shows the plot (needed for pycharm)

