import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

# load the dataset
data = load_breast_cancer()

# praper x and y
X = data.data[:,:2]
y = data.target 

# scale the data 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# spliting the data for traing and testing
X_train,X_test ,y_train,  y_test = train_test_split(X,y, test_size=0.33, random_state=42, shuffle=True)

# difining the positive and negative values
positive = X[y==1]
negative = X[y==0]

# ploting the positive 
fig,ax = plt.subplots(figsize = (8,8))
ax.scatter(positive[:,0],positive[:,1],  marker='o', s=50, c='b')
ax.scatter(negative[:,0],negative[:,1],  marker='x', s=50, c='r')
ax.set_xlabel(data.feature_names[0])
ax.set_ylabel(data.feature_names[1])
ax.legend()
plt.show()

# calling the logistic regression class
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

# checking the accuracy for training and testing data
score_train = LogReg.score(X_train, y_train)
print('training accuracy: ',score_train*100,'%')

score_test = LogReg.score(X_test, y_test)
print('testing accuracy: ',score_test*100 ,'%')

# predicting the values for tesring data
y_pred = LogReg.predict(X_test)
print("predicted y values: ",y_pred[:30])
print("testing y values:   ",y_test[:30])

# calling the confusion matrix
con_mat = confusion_matrix(y_test,y_pred)
print(con_mat)

# ploting the confusion matrix
sns.heatmap(con_mat, center=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotting the decision boundary
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(positive[:, 0], positive[:, 1], marker='o', s=50, c='b', label='Positive')
ax.scatter(negative[:, 0], negative[:, 1], marker='x', s=50, c='r', label='Negative')

# Create a mesh grid for the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the label for each point in the mesh grid
Z = LogReg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
ax.set_xlabel(data.feature_names[0])
ax.set_ylabel(data.feature_names[1])
ax.legend()
plt.show()




