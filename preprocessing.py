import pandas as pd
import numpy as np
import pickle
import Parameters as param
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
    print('Exporting prescaled pickle files ...')
    with open('./supporting_data_files/Max_array' + text + '.pkl', 'wb') as e:
        pickle.dump(max_array, e)
    with open('./supporting_data_files/Min_array' + text + '.pkl', 'wb') as f:
        pickle.dump(min_array, f)


def main():
    # load pickle files
    input_data = pd.read_pickle(open('./supporting_data_files/input_data_frame.pkl', 'rb')).T
    output_data = pd.read_pickle(open('./supporting_data_files/output_data_frame.pkl', 'rb'))

    input_data = input_data[['temperature', 'velocity']]
    output_data = output_data[['temperature', 'velocity']]

    # export scaling matricies
    output_matrix = np.asmatrix(output_data)
    min_max_export(output_matrix, 'output')

    print()
    print('Fetching data frames ...')
    print()
    print("This may take a while!")

    design_points = len(output_data)

    # fit and transform minmax scaler values
    minmax_scale_input = preprocessing.MinMaxScaler().fit(input_data)
    minmax_input = minmax_scale_input.transform(input_data).T

    minmax_scale_output = preprocessing.MinMaxScaler().fit(output_data)
    minmax_output = minmax_scale_output.transform(output_data)

    # restore data frame structure after scaling
    input_velocity_df = array_to_df(len(minmax_input) // 2, 0, minmax_input, design_points, 'dp{}') \
        .rename(columns={'1': 'main_inlet_velocity', '2': 'sec_inlet_velocity'})
    input_temperature_df = array_to_df(len(minmax_input) // 2, 2, minmax_input, design_points, 'dp{}') \
        .rename(columns={'1': 'main_inlet_temperature', '2': 'sec_inlet_temperature'})

    scaled_input_df = pd.concat({'velocity': input_velocity_df, 'temperature': input_temperature_df}, axis=1)

    output_velocity_df = array_to_df(len(minmax_output.T) // 2, 5000, minmax_output.T, design_points, 'dp{}')
    output_temperature_df = array_to_df(len(minmax_output.T) // 2, 0, minmax_output.T, design_points, 'dp{}')

    print()
    print('splitting data sets ...')

    scaled_output_df = pd.concat({'velocity': output_velocity_df, 'temperature': output_temperature_df}, axis=1)

    # split testing and training data sets
    X_train, X_test, Y_train, Y_test = train_test_split(scaled_input_df, scaled_output_df, test_size=0.2,
                                                        random_state=42)
    print()
    print("Input training set:")
    print(X_train)
    print()
    print("Output training set:")
    print(Y_train)
    print()
    print("Input testing set:")
    print(X_test)
    print()
    print("Output testing set:")
    print(Y_test)

    # export pickle files of scalar values and datasets
    output_data.to_pickle('./supporting_data_files/Actual_Output.pkl')
    scaled_output_df.to_pickle('./supporting_data_files/output_scalar_df.pkl')
    scaled_input_df.to_pickle('./supporting_data_files/input_scalar_df.pkl')
    X_train.to_pickle('./supporting_data_files/input_training_set.pkl')
    X_test.to_pickle('./supporting_data_files/input_testing_set.pkl')
    Y_train.to_pickle('./supporting_data_files/output_training_set.pkl')
    Y_test.to_pickle('./supporting_data_files/output_testing_set.pkl')

    print()
    print("Data preprocessing complete!")

if __name__ == '__main__':
    main()
    print()