from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import numpy as np


#load training and testing data samples
x_train_df = pd.read_pickle(open('./supporting_data_files/input_training_set.pkl', 'rb'))
y_train_df = pd.read_pickle(open('./supporting_data_files/output_training_set.pkl', 'rb'))

x_test_df = pd.read_pickle(open('./supporting_data_files/input_testing_set.pkl', 'rb'))
y_test_df = pd.read_pickle(open('./supporting_data_files/output_testing_set.pkl', 'rb'))

#map data frames to numpy arrays
x_train, y_train = x_train_df.to_numpy(), y_train_df.to_numpy()
x_test, y_test = x_test_df.to_numpy(), y_test_df.to_numpy()


#init random forest structure
accumulated_error = []
accumulated_score = []

print('Training ...')
regressor = RandomForestRegressor(n_estimators=150, random_state=42)
# train random forrest
regressor.fit(x_train_df, y_train_df)

# evaluate training results with testing set
print('Testing ...')
y_hyp = regressor.predict(x_test_df)

error = mean_squared_error(y_test, y_hyp)
mean = np.mean(y_test)
score = 100 * error / mean

print('Exporting pickle ...')
with open('./supporting_data_files/Random_Forest.pkl', 'wb') as f:
    pickle.dump(regressor, f)
print('Done!')
