from sklearn.linear_model import LinearRegression
import numpy as np
house_sizes = np.array([1500, 2000, 2500])
prices = np.array([200000, 250000, 300000])  # Assuming corresponding prices
# Reshape to 2D arrays
house_sizes_reshaped = house_sizes.reshape(-1, 1) #(n_samples, 1) -1 transpose 
# Train the model
model = LinearRegression()
model.fit(house_sizes_reshaped, prices)
# Predict price for a new house size (example)
prediction = model.predict([[40000]])[0]
print("Predicted price for 40000 sqft house:", prediction)