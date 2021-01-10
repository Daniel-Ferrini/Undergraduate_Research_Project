########################################################################################################################
###############################################    Project Variables    ################################################
########################################################################################################################

root_path = '/Undergraduate_Research_Project'
                                                                            # indicate path to cloned repository



Training_Status = 'Evaluation'
                                                                            # indicate operating mode of program
                                                                            # accepts: 'Training' or 'Evaluation'
                                                                            # (default set to evaluation mode)
Data_Range = 'Original'
                                                                            # indicate CFD evaluation data range
                                                                            # accepts: 'Original',
                                                                            # 'Lower-bound', or 'Upper-bound'
ML_Model = 'Random Forest'
                                                                            # indicate desired ML model
                                                                            # accepts: 'Neural Network',
                                                                            # 'Random Forest', or 'Autoencoder'
Fluid_Property = 'Temperature'
                                                                            # indicate fluid metric
                                                                            # accepts: 'Temperature',
                                                                            # or 'Velocity'

########################################################################################################################
########################################################################################################################
########################################################################################################################

def secondary_location(training_status, file_location):
    path = ""
    index = 0
    try:
        if training_status == 'Evaluation':
            try:
                if file_location == 'Original':
                    path = "/CFD (1000 dp)_files/"
                elif file_location == 'Lower-bound':
                    path = "/Lower-bound_data/"
                    index = 1020
                elif file_location == 'Upper-bound':
                    path = "/Upper_bound_data/"
                    index = 1040
                return path, index
            except UnboundLocalError as error:
                return error

        elif training_status == 'Training':
            path = "/CFD (1000 dp)_files/"
            return path, index

    except UnboundLocalError as error:
        return error


# path exporter
if type(secondary_location(Training_Status, Data_Range)[0]) == str:
    path = root_path + secondary_location(Training_Status, Data_Range)[0]

else:
    path = UnboundLocalError
    print(path)

index = secondary_location(Training_Status, Data_Range)[1]

if Fluid_Property == "Temperature":
    metric = " (K)"
else:
    metric = " (m/s)"

print()
print("=================================================================================")
print("===========================   Selected Parameters     ===========================")
print("=================================================================================")
print()
print("root_path --> " + root_path)
print()
print("Training_Status --> " + Training_Status)
print()
print("Data_Range --> " + Data_Range)
print()
print("ML_Model --> " + ML_Model)
print()
print("Fluid_Property --> " + Fluid_Property)
print()
print("=================================================================================")
print("=================================================================================")
print("=================================================================================")
print()


