import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# data frame reinitialisation functions
def array_to_df(number_of_coulumns, starting_point, data, x, row_str):
    scaled_output_df = pd.DataFrame()
    for row in range(number_of_coulumns):
        index = 1 + row
        string = "{}".format(index)
        column = {string: data[starting_point + row]}
        new_df = pd.DataFrame(column)
        scaled_output_df = pd.concat([scaled_output_df, new_df], axis=1)

    cellnumber = create_index(x, row_str)
    scaled_output_df = pd.concat([scaled_output_df, cellnumber], axis=1)

    return scaled_output_df.set_index('cellnumber')

def create_index(x, row_str):
    new_index = []
    for i in range(1, x+1, 1):
        format = row_str.format(i)
        new_index.append(format)

    new_index_df = pd.DataFrame({'cellnumber': new_index})

    return new_index_df

# export max & min arrays
def min_max_export(origional_array, string):
    origional_transpose = origional_array.T
    max_array = np.array([])
    min_array = np.array([])
    for i in range(len(origional_transpose)):
        max = np.max(origional_transpose[i])
        min = np.min(origional_transpose[i])
        max_array = np.append(max_array, max)
        min_array = np.append(min_array, min)

    text = "_{}".format(string)
    print('exporting pickle ...')
    with open('Max_array' + text + '.pickle', 'wb') as e:
        pickle.dump(max_array, e)
    with open('Min_array' + text + '.pickle', 'wb') as f:
        pickle.dump(min_array, f)


# load pickle files
# input_data = pd.DataFrame(pickle.load(open('input_data_frame.pickle', 'rb'))).T
# output_data = pd.DataFrame(pickle.load(open('output_data_frame.pickle', 'rb')))
input_data = pd.read_pickle(open('input_data_frame.pickle', 'rb')).T
output_data = pd.read_pickle(open('output_data_frame.pickle', 'rb'))

input_data = input_data[['temperature', 'velocity']]
output_data = output_data[['temperature', 'velocity']]

print(input_data)
print(output_data)

# export scaling matricies
inpurt_matrix = np.asmatrix(input_data)
output_matrix = np.asmatrix(output_data)
min_max_export(output_matrix, 'output')

print('fetching data frames ...')

design_points = len(output_data)

# fit and transform minmax scaler values
minmax_scale_input = preprocessing.MinMaxScaler().fit(input_data)
minmax_input = minmax_scale_input.transform(input_data).T

minmax_scale_output = preprocessing.MinMaxScaler().fit(output_data)
minmax_output = minmax_scale_output.transform(output_data)

# restore data frame structure after scaling
input_velocity_df = array_to_df(len(minmax_input)//2, 0, minmax_input, design_points, 'dp{}')\
    .rename(columns={'1':'main_inlet_velocity', '2':'sec_inlet_velocity'})
input_temperature_df = array_to_df(len(minmax_input)//2, 2, minmax_input, design_points, 'dp{}')\
    .rename(columns={'1':'main_inlet_temperature', '2':'sec_inlet_temperature'})

# input_velocity_df = array_to_df(len(minmax_input)//2, 2, minmax_input, design_points, 'dp{}')\
#     .rename(columns={'1':'main_inlet_velocity', '2':'sec_inlet_velocity'})
# input_temperature_df = array_to_df(len(minmax_input)//2, 0, minmax_input, design_points, 'dp{}')\
#     .rename(columns={'1':'main_inlet_temperature', '2':'sec_inlet_temperature'})

scaled_input_df = pd.concat({'velocity': input_velocity_df, 'temperature': input_temperature_df}, axis=1)

output_velocity_df = array_to_df(len(minmax_output.T)//2, 5000, minmax_output.T, design_points, 'dp{}')
output_temperature_df = array_to_df(len(minmax_output.T)//2, 0, minmax_output.T, design_points, 'dp{}')

# output_velocity_df = array_to_df(len(minmax_output.T)//2, 0, minmax_output.T, design_points, 'dp{}')
# output_temperature_df = array_to_df(len(minmax_output.T)//2, 5000, minmax_output.T, design_points, 'dp{}')

print('splitting data sets ...')

scaled_output_df = pd.concat({'velocity': output_velocity_df, 'temperature': output_temperature_df}, axis=1)

# split testing and training data sets
X_train, X_test, Y_train, Y_test = train_test_split(scaled_input_df, scaled_output_df, test_size=0.2, random_state=42)


print(X_train, Y_train)
print(X_test, Y_test)


# export pickle files of scaler values and datasets
output_data.to_pickle('Actual_Output.pickle')
scaled_output_df.to_pickle('output_scalar_df.pickle')
scaled_input_df.to_pickle('input_scalar_df.pickle')
X_train.to_pickle('input_training_set.pickle')
X_test.to_pickle('input_testing_set.pickle')
Y_train.to_pickle('output_training_set.pickle')
Y_test.to_pickle('output_testing_set.pickle')
