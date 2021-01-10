import pandas as pd
import Parameters as param
import os
import pickle

#sort folder list
def reorgenise_folder(list):
    folder_list = []
    new_folder_index = []
    sorted_folder_list = []
    for folder in list:
        counter = folder.lstrip('dp')
        file_num = int(counter) - 1
        new_folder_index.append(file_num - param.index)
        folder_list.append(folder)
        sorted_folder_list.append(0)
    #assign desig point values to new indecies
    for i in range(len(new_folder_index)):
        sorted_folder_list[new_folder_index[i]] = folder_list[i]
    return sorted_folder_list

#create new sorted data frame
def restructure_array(input_array, index_array):
    new_df = pd.DataFrame()
    for design_point in range(len(input_array)):
        new_df = new_df.append(input_array.loc[index_array[design_point], :])
    return new_df

def load_instance_data(path):
    # empty array and data frame initialisation
    main_vel_list = []
    sec_vel_list = []
    main_temp_list = []
    sec_temp_list = []
    input_main_velocity = pd.DataFrame()
    input_sec_velocity = pd.DataFrame()
    input_main_temperature = pd.DataFrame()
    input_sec_temperature = pd.DataFrame()

    # read and extract .csv data
    df = pd.read_csv(path + '/DesignPointLog.csv', sep=';', index_col='Design Point')
    main_vel_list.append(df['main_inlet_velocity [m s^-1]'])
    sec_vel_list.append(df['sec_inlet_velocity [m s^-1]'])
    main_temp_list.append(df['main_inlet_temperature [K]'])
    sec_temp_list.append(df['sec_inlet_temperature [K]'])

    # insert lists into seperate data frames
    input_main_velocity = input_main_velocity.append(main_vel_list)
    input_sec_velocity = input_sec_velocity.append(sec_vel_list)
    input_main_temperature = input_main_temperature.append(main_temp_list)
    input_sec_temperature = input_sec_temperature.append(sec_temp_list)
    input_velocity_array = input_main_velocity.append(input_sec_velocity)
    input_temperature_array = input_main_temperature.append(input_sec_temperature)

    # restructure data frames
    input_data_frame = pd.concat({'velocity': input_velocity_array,
                                  'temperature': input_temperature_array})

    print()
    print("Input Data Frame:")
    print(input_data_frame)
    print()

    # export as pickle file
    input_data_frame.to_pickle('./supporting_data_files/input_data_frame.pkl')

def load_output_data(path):
    # empty array and data frame initialisation
    index = 0
    folder_list = []
    vel_list = []
    temp_list = []
    xcoord_list = pd.DataFrame()
    ycoord_list = pd.DataFrame()
    velocity_array = pd.DataFrame()
    temperature_array = pd.DataFrame()

    for folder in os.listdir(path):
        try:
            if folder != ('DATAFILE'):
                # load CFD data file as pandas dataframe
                df = pd.read_csv(path + str(folder) + '/FLU/Fluent/DATAFILE',
                                 delim_whitespace=True, index_col='cellnumber')
                new_index = ("{}".format(folder))
                xcoord_list = xcoord_list.append(df['x-coordinate'])
                ycoord_list = ycoord_list.append(df['y-coordinate'])
                # rename columns to correspond to the design points from which they were sourced
                df_new_velocity = df.rename(columns={'velocity-magnitude': new_index})
                df_new_temperature = df.rename(columns={'temperature': new_index})
                # sort and append values to independent data frames
                vel_list.append(df_new_velocity)
                temp_list.append(df_new_temperature)
                velocity_array = velocity_array.append(vel_list[index][new_index])
                temperature_array = temperature_array.append(temp_list[index][new_index])
                index += 1
                folder_list.append(folder)

        except IOError:
            continue

    # restructure data frames
    velocity_array = restructure_array(velocity_array, reorgenise_folder(folder_list))
    temperature_array = restructure_array(temperature_array, reorgenise_folder(folder_list))

    # create multi-index pandas dataframe where the velocity and temperature magnituudes
    # at each cell are related to the corresponding design points
    output_data_frame = pd.concat({'velocity': velocity_array, 'temperature': temperature_array}, axis=1)
    coordinate_data_frame = pd.concat([xcoord_list.drop_duplicates(),
                                       ycoord_list.drop_duplicates()],
                                      axis=0, join='inner')

    print()
    print("Output Data Frame:")
    print(output_data_frame)

    print()
    print("Coordinate Data Frame:")
    print(coordinate_data_frame)

    # export as pickle file
    output_data_frame.to_pickle("./supporting_data_files/output_data_frame.pkl")
    coordinate_data_frame.to_pickle("./supporting_data_files/coordinate_data_frame.pkl")

def main():
    if param.path != UnboundLocalError:
        print("Loading instance data...")
        load_instance_data(param.path)
        print("Loading simulation data...")
        load_output_data(param.path)
        print("CFD data has successfully loaded")
    else:
        print("Failed to compile program, please correct syntax in Parameters.py")

if __name__ == '__main__':
    main()
    print()
