import torch
import pickle
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib import gridspec
import timeit
import pandas as pd
import resource
import os
import Parameters as param

# rescaling function
def rescale(scaled_array, max_array, min_array):
    domain = max_array - min_array
    rescaled_array = np.empty([1, domain.size])
    for j in range(len(scaled_array)):
        row_change = np.array([])
        for a in range(domain.size):
            row_change = np.append(row_change, domain[a] * scaled_array[j, a])
        relative_change = row_change + min_array
        rescaled_array = np.vstack((rescaled_array, relative_change))

    rescaled_array = np.delete(rescaled_array, 0, axis=0)
    return rescaled_array


# mean array function
def mean_array(array):
    transpose = array.T
    mean = np.array([])
    for i in range(len(transpose)):
        mean = np.append(mean, np.mean(transpose[i]))

    return mean


def init_model(ML_Model, input_array, output_array):
    # model exporter
    if ML_Model == "Neural Network":
        return NN_init(input_array, output_array)

    elif ML_Model == "Random Forest":
        return RF_init(input_array)

    elif ML_Model == "Autoencoder":
        return AE_init(input_array, output_array)

    else:
        print("Failed to compile program, please correct ML_Model syntax in Parameters.py")


def NN_init(input_array, output_array):
    from model_training_files import NN
    import matplotlib
    matplotlib.use('TkAgg')

    # load neural network class
    neural_network = NN

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialise mlp structures
    NN_struct = neural_network.NN(input_dim=input_array.shape[1], output_dim=output_array.shape[1],
                                  n_hidden=128)

    # upload trained algorithim states
    NN_struct.load_state_dict(torch.load('./supporting_data_files/NN.pt',
                                         map_location=torch.device('cpu')))
    NN_struct.to(device)
    NN_struct.eval()
    instance_tensor = torch.from_numpy(input_array).to(device)

    return instance_tensor, NN_struct, device


def RF_init(input_array):
    # load pickle files
    regressor = pd.read_pickle(open('./supporting_data_files/Random_Forest.pkl', 'rb'))

    return input_array, regressor

def AE_init(input_array, output_array):
    from model_training_files import Autoencoder
    from model_training_files import Neural_Network

    # declare imported files as variables
    neural_network = Neural_Network
    autoencoder = Autoencoder

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers = [512, 128, 64, 32]

    # initialise mlp structures
    nn_struct = neural_network.NN(input_dim=input_array.shape[1], output_dim=layers[3],
                                  n_hidden=512)
    ae_struct = autoencoder.Autoencoder(visible_dim=output_array.shape[1], coding_dim1=layers[0],
                                        coding_dim2=layers[1], coding_dim3=layers[2], coding_dim4=layers[3])

    # upload trained algorithim states
    nn_struct.load_state_dict(torch.load('./supporting_data_files/NN_512.pt',
                                         map_location=torch.device('cpu')))
    ae_struct.load_state_dict(torch.load('./supporting_data_files/Autoencoder_512.pt',
                                         map_location=torch.device('cpu')))

    nn_struct = nn_struct.to(device)
    ae_struct = ae_struct.to(device)
    nn_struct.eval()
    ae_struct.eval()

    input_tensor = torch.from_numpy(input_array).to(device)

    return input_tensor, nn_struct, ae_struct, device


def contour_plotter(output, prediction_error):
    # plot mean and error
    df = pickle.load(open('./supporting_data_files/coordinate_data_frame.pkl', 'rb'))
    pos_data = (np.asmatrix(df.iloc[:, :]))

    x_pos = np.array(pos_data[0, 0:5000]).flatten()
    y_pos = np.array(pos_data[1, 0:5000]).flatten()

    # prediction plot
    fig = plt.figure(0, figsize=(15, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=1,
                             width_ratios=[4, 1, 4])
    plt.subplot(spec[0])
    rect_top = plt.Rectangle((0.0025, 0.0975), 0.5, 0.4, fc='white')
    rect_bottom = plt.Rectangle((0.0025, 0.025), 0.55, -0.5225, fc='white')
    plt.gca().add_patch(rect_top)
    plt.gca().add_patch(rect_bottom)
    plt.tricontour(x_pos, y_pos, output, 200, linewidths=0, colors='k')
    plt.tricontourf(x_pos, y_pos, output, 200, cmap=plt.cm.jet)
    plt.title('Mean ' + param.Fluid_Property + ' Prediction Contour for ' + param.ML_Model)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    clb = plt.colorbar()
    clb.set_label(param.Fluid_Property + param.metric)

    # error plot
    plt.subplot(spec[2])
    rect_top = plt.Rectangle((0.0025, 0.0975), 0.5, 0.4, fc='white')
    rect_bottom = plt.Rectangle((0.0025, 0.025), 0.55, -0.5225, fc='white')
    plt.gca().add_patch(rect_top)
    plt.gca().add_patch(rect_bottom)
    plt.tricontour(x_pos, y_pos, prediction_error, 200, linewidths=0, colors='k')
    plt.tricontourf(x_pos, y_pos, prediction_error, 200, cmap=plt.cm.jet)
    plt.title('Mean ' + param.Fluid_Property + ' Error Contour for ' + param.ML_Model)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    clb = plt.colorbar()
    clb.set_label('Error (%)')

    plt.show()

def hist_plotter(error_mean):
    # histogram
    t_data_error, v_data_error = np.hsplit(error_mean, 2)

    temp_e = t_data_error.flatten()
    vel_e = v_data_error.flatten()

    t_data_error, v_data_error = np.hsplit(error_mean, 2)
    temp_err = t_data_error.flatten()
    vel_err = v_data_error.flatten()

    print("Mean Temperature Error = {:.3f}%, Max Temperature Error = {:.3f}%"
          .format(np.mean(temp_e), np.max(temp_e)))
    print("Mean " + param.Fluid_Property + " Error = {:.3f}%, Max " + param.Fluid_Property + " Error = {:.3f}%"
          .format(np.mean(vel_e), np.max(vel_e)))
    n_bins = 30
    cm = plt.cm.jet

    fig = plt.figure(1, figsize=(15, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=1,
                             width_ratios=[4, 1, 4])
    plt.subplot(spec[0])
    plt.title('Temperature Error Distribution for ' + param.ML_Model)
    plt.xlabel('Error Range (%)')
    plt.ylabel('Frequency')
    (counts, bins) = np.histogram(temp_err, n_bins)
    factor = 0.00002
    N, bins, patches = plt.hist(bins[:-1], bins, weights=counts * factor)
    plt.yscale('log')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.subplot(spec[2])
    plt.title('Velocity Error Distribution for ' + param.ML_Model)
    plt.xlabel('Error Range (%)')
    plt.ylabel('Frequency')
    (counts, bins) = np.histogram(vel_err, n_bins)
    factor = 0.00002
    N, bins, patches = plt.hist(bins[:-1], bins, weights=counts * factor)
    plt.yscale('log')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.show()

def main():
    print("Generating prediction contours...")
    print()

    # load pickle files
    instance_array = np.asmatrix(pickle.load(open('./supporting_data_files/input_scalar_df.pkl', 'rb')))
    actual_output = np.asmatrix(pickle.load(open('./supporting_data_files/Actual_Output.pkl', 'rb')))

    # open origional df
    max_output_array = pickle.load(open('./supporting_data_files/Max_array_output.pkl', 'rb'))
    min_output_array = pickle.load(open('./supporting_data_files/Min_array_output.pkl', 'rb'))

    model_params = init_model(param.ML_Model, instance_array, actual_output)

    start = timeit.default_timer()
    # regressor predictions
    memoryUse_a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if param.ML_Model == "Neural Network":
        prediction = model_params[1](model_params[0].float())
        prediction = prediction.detach().cpu().numpy()
    elif param.ML_Model == "Random Forest":
        prediction = model_params[1](model_params[0])
    else:
        prediction = model_params[1](model_params[0].float())
        prediction = model_params[2].decoder(prediction)
        prediction = prediction.detach().cpu().numpy()

    # create rescaled arrays
    rescaled_prediction = rescale(prediction, max_output_array, min_output_array)
    memoryUse_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stop = timeit.default_timer()

    total_time = stop - start
    total_memory = (memoryUse_b - memoryUse_a) * (10 ** -6)
    print("Run time = {:.3f} s".format(total_time))
    print("memory use = {:.3f} MB".format(total_memory))
    print()

    # calculate error matrix
    error = rescaled_prediction[0] - actual_output[0]
    for row in range(len(rescaled_prediction) - 1):
        error = np.vstack((error, rescaled_prediction[row + 1] - actual_output[row + 1]))

    error = (np.divide(np.absolute(error), actual_output)) * 100
    prediction_mean = mean_array(rescaled_prediction)
    error_mean = mean_array(error)

    # split mean and error arrays
    t_data_prediction, v_data_prediction = np.hsplit(prediction_mean, 2)
    t_data_error, v_data_error = np.hsplit(error_mean, 2)

    if param.Fluid_Property == "Temperature":
        output = np.array(t_data_prediction).flatten()
        prediction_error = np.array(t_data_error).flatten()
    elif param.Fluid_Property == "Velocity":
        output = np.array(v_data_prediction).flatten()
        prediction_error = np.array(v_data_error).flatten()
    else:
        print("Incorrect fluid property specified, please use correct syntax in Parameters.py")

    contour_plotter(output, prediction_error)
    hist_plotter(error_mean)

if __name__ == '__main__':
    if param.Training_Status == "Evaluation":
        main()
    elif param.Training_Status == "Training":
        os.system('python ./model_training.py')
    else:
        print("Failed to compile program, please indicate correct Training_Status in Parameters.py")

    print()
