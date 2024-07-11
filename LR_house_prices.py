# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Generating synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Random house sizes
#X = np.random.rand(100, 1, low=0, high=2)
y = 50 + 30 * X + np.random.randn(100, 1) * 10  
# Price = 50 + 30 * size + noise
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                    random_state=0)
# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)

# Plotting the training data, testing data, and the regression line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.plot(X_test, y_pred, color='green', label='Regression Line')
plt.xlabel('House Size (sq. ft)')
plt.ylabel('House Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.show()
