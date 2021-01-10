import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

#create neural network class
class NN(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden):
        super(NN, self).__init__()
        # initialise number input variables, output variables,
        # and number of neurons per hidden layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

        #set number of conections between layers
        self.fc1 = nn.Linear(self.input_dim, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden,self.output_dim)

    #define forward propogation calculations
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

#create mse cost fucntion for validation
def cost_function(y_hyp, y):
    return F.mse_loss(y_hyp, y, reduction='mean')

def main():
    # load pickle files and split encoding space
    encoding_space = pickle.load(open('./supporting_data_files/Encoding_Space.pkl', 'rb'))
    Y_train, Y_test = train_test_split(encoding_space, test_size=0.2, shuffle=False)

    x_train_df = pickle.load(open('./supporting_data_files/input_training_set.pkl', 'rb'))
    x_test_df = pickle.load(open('./supporting_data_files/input_testing_set.pkl', 'rb'))

    # map numpy arrays to torch tensors
    x_train_tensor = torch.from_numpy(x_train_df.to_numpy())
    y_train_tensor = torch.from_numpy(Y_train)

    x_test_tensor = torch.from_numpy(x_test_df.to_numpy())
    y_test_tensor = torch.from_numpy(Y_test)

    # create dataset
    training_ds = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    testing_ds = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # create dataloader from dataset
    training_dl = torch.utils.data.DataLoader(training_ds, shuffle=False)
    testing_dl = torch.utils.data.DataLoader(testing_ds, shuffle=False)

    # init GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init neural network structure
    NN_struct = NN(x_train_df.to_numpy().shape[1], encoding_space.shape[1], n_hidden=16)
    NN_struct.to(device)

    optimizer = optim.Adam(NN_struct.parameters(), lr=0.001)

    # train NN
    training_error = []
    testing_error = []

    for epoch in range(2):
        NN_struct.train()
        running_loss = 0
        # loop over training set
        for x, y in training_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hyp = NN_struct(x.float())
            error = cost_function(y_hyp, y.float())
            error.backward()
            optimizer.step()
            running_loss += error.item()

        # evaluate training results with testing set
        NN_struct.eval()
        validation_error = 0
        with torch.no_grad():
            validation_error = sum(cost_function(NN_struct(x.to(device).float()),
                                                 y.to(device).float()) for x, y in testing_dl)

        # prints current training status to console
        if epoch % 1 == 0:
            print('Epoch: %d, Error: %.3f' %
                  (epoch + 1, running_loss))

        # creates error entries for each iteration
        training_error.append(running_loss / len(testing_dl.dataset))
        testing_error.append(validation_error / len(testing_dl.dataset))

    mean = np.mean(Y_test)
    score = 100 * error.item() / mean

    print("Error = %.3f , Score = %.3f" % (error, score))

    # plot training and testing results for given NN structure
    plt.plot(training_error, 'orange')
    plt.plot(testing_error, 'blue')
    plt.yscale('log')
    plt.legend(['Training set', 'Validation set'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.show()

    print('Exporting files ...')
    torch.save(NN_struct.state_dict(), './supporting_data_files/NN_512.pt')
    print('Done!')

if __name__ == '__main__':
    main()
