import Parameters as param
import os


def main():

    if param.Data_Range != 'Original':
        param.Data_Range = 'Original'
        os.system('python ./data_loader.py')
        os.system('python ./preprocessing.py')

    if param.Training_Status == "Training":
        if param.ML_Model == "NN":
            os.system('python ./model_training_files/NN.py')
        elif param.ML_Model == "Random Forest":
            os.system('python ./model_training_files/Random_Forest.py')
        elif param.ML_Model == "Autoencoder":
            print("Training Autoencoder...")
            print()
            os.system('python ./model_training_files/Autoencoder.py')
            print()
            print("Training Neural Network...")
            print()
            os.system('python ./model_training_files/Neural_Network.py')
        else:
            print("Failed to compile program, please indicate correct ML_model in Parameters.py")

    else:
        print("Failed to compile program, please indicate correct Training_Mode in Parameters.py")

if __name__ == '__main__':
    main()
    print()
