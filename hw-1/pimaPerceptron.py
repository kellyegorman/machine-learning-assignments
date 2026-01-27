# Assignment:

# Train a Perceptron to the entire Pima Indians data set (Available at https://www.kaggle.
# com/datasets/kumargh/pimaindiansdiabetescsv/data). Take a look at the data card to
# understand the dataset. Do not split the data into training and test for this problem!
# 1. What are the input features and the output classes? (2 points)
# 2. Experiment with different learning rates and report the highest (training set) classifi-
# cation accuracy you can obtain. (10 points)
# 3. For the learning rate that gives the highest classification accuracy, plot the number of
# misclassification errors against the number of epochs, similar to Figure 2.7 from the
# textbook. (8 points)

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('hw-1/pima-indians-diabetes.csv')
print(df.head())

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = Perceptron(random_state=1, eta0=0.01, max_iter=1000)
model.fit(X, y)

predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# 1. 
# Input -> Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Output -> Class (0 or 1 if they have diabetes)

# 2. 
# The highest training set classification accuracy I found was 74.48% 
# with a random state of 1 and learning rate of 0.01.

# 3.
errors = []
for epoch in range(1, 101):
    model = Perceptron(random_state=1, eta0=0.01, max_iter=epoch, tol=None)
    model.fit(X, y)
    predictions = model.predict(X)
    misclassified = (y != predictions).sum()
    errors.append(misclassified)
plt.plot(range(1, 101), errors, marker='o')
plt.xlabel('Number of epochs')
plt.ylabel('Number of misclassification errors')
plt.title('Misclassification errors vs epochs')
plt.grid()
plt.show()
