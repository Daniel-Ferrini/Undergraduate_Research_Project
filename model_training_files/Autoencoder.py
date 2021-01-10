import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

#create Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, visible_dim, coding_dim1, coding_dim2,
                 coding_dim3, coding_dim4):
        super(Autoencoder, self).__init__()
        # initialise number input variables, output variables,
        # and dimensionality of encoding space
        self.visible_dim = visible_dim
        self.coding_dim1 = coding_dim1
        self.coding_dim2 = coding_dim2
        self.coding_dim3 = coding_dim3
        self.coding_dim4 = coding_dim4
        self.dropout = nn.Dropout(p=0.5)

        #encoder structure
        self.enc1 = nn.Linear(self.visible_dim, self.coding_dim1)
        self.enc2 = nn.Linear(self.coding_dim1, self.coding_dim2)
        self.enc3 = nn.Linear(self.coding_dim2, self.coding_dim3)
        self.enc4 = nn.Linear(self.coding_dim3, self.coding_dim4)

        self.dec1 = nn.Linear(self.coding_dim4, self.coding_dim3)
        self.dec2 = nn.Linear(self.coding_dim3, self.coding_dim2)
        self.dec3 = nn.Linear(self.coding_dim2, self.coding_dim1)
        self.dec4 = nn.Linear(self.coding_dim1, self.visible_dim)

        self.bn1 = nn.BatchNorm1d(num_features=self.coding_dim1)
        self.bn2 = nn.BatchNorm1d(num_features=self.coding_dim2)
        self.bn3 = nn.BatchNorm1d(num_features=self.coding_dim3)
        self.bn4 = nn.BatchNorm1d(num_features=self.coding_dim4)

    def encoder(self, enc):
        enc = self.dropout(F.relu(self.bn1(self.enc1(enc))))
        enc = self.dropout(F.relu(self.bn2(self.enc2(enc))))
        enc = self.dropout(F.relu(self.bn3(self.enc3(enc))))
        enc = F.relu(self.bn4(self.enc4(enc)))
        return enc

    def decoder(self, dec):
        dec = F.relu(self.dec1(dec))
        dec = F.relu(self.dec2(dec))
        dec = F.relu(self.dec3(dec))
        dec = F.relu(self.dec4(dec))
        return dec

    #define forward propogation calculations
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#create mse cost fucntion for validation
def cost_function(y_hyp, y):
    return F.mse_loss(y_hyp, y, reduction='mean')

def main():
    # load training and testing data samples
    x_train_df = pickle.load(open('./supporting_data_files/input_training_set.pkl', 'rb'))
    y_train_df = pickle.load(open('./supporting_data_files/output_training_set.pkl', 'rb'))

    x_test_df = pickle.load(open('./supporting_data_files/input_testing_set.pkl', 'rb'))
    y_test_df = pickle.load(open('./supporting_data_files/output_testing_set.pkl', 'rb'))

    # map numpy arrays to torch tensors
    x_train_tensor = torch.from_numpy(x_train_df.to_numpy())
    y_train_tensor = torch.from_numpy(y_train_df.to_numpy())

    x_test_tensor = torch.from_numpy(x_test_df.to_numpy())
    y_test_tensor = torch.from_numpy(y_test_df.to_numpy())

    output_data = np.vstack((y_train_df.to_numpy(), y_test_df.to_numpy()))

    # create dataset
    training_ds = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    testing_ds = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # create dataloader from dataset
    training_dl = torch.utils.data.DataLoader(training_ds, batch_size=8, shuffle=False)
    testing_dl = torch.utils.data.DataLoader(testing_ds, batch_size=8, shuffle=False)

    # init GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers = [512, 128, 64, 32]
    accumulated_error = []
    accumulated_score = []

    # init Autoencoder structure
    autoencoder = Autoencoder(visible_dim=y_train_df.to_numpy().shape[1], coding_dim1=layers[0],
                              coding_dim2=layers[1], coding_dim3=layers[2], coding_dim4=layers[3])

    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # train Autoencoder
    training_error = []
    testing_error = []

    for epoch in range(500):
        autoencoder.train()
        running_loss = 0
        # loop over training set
        for x, y in training_dl:
            y = y.to(device)
            optimizer.zero_grad()
            y_hyp = autoencoder(y.float())
            error = cost_function(y_hyp, y.float())
            error.backward()
            optimizer.step()
            running_loss += error.item()

        # evaluate training results with testing set
        autoencoder.eval()
        validation_error = 0
        with torch.no_grad():
            y_pred = autoencoder(y.to(device).float())
            validation_error = sum(cost_function(y_pred,
                                                 y.to(device).float()) for x, y in testing_dl)

        # prints current training status to console
        if epoch % 1 == 0:
            print('Epoch: %d, Error: %.3f' %
                  (epoch + 1, running_loss))

        # creates error entries for each iteration
        training_error.append(running_loss / len(testing_dl.dataset))
        testing_error.append(validation_error / len(testing_dl.dataset))

    encoding_dimention = autoencoder.encoder(torch.from_numpy(output_data).float())

    mean = np.mean(y_test_df.to_numpy())
    score = 100 * error.item() / mean

    accumulated_error.append(error)
    accumulated_score.append(score)

    print("Error = %.3f , Score = %.3f" % (error, score))

    # plot training and testing results for given Autoencoder structure
    plt.plot(training_error, 'orange')
    plt.plot(testing_error, 'blue')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend(['Training set', 'Validation set'])
    plt.tight_layout()
    plt.show()

    print('Exporting pickle ...')
    with open('./supporting_data_files/Encoding_Space.pkl', 'wb') as f:
        pickle.dump(encoding_dimention, f)
    torch.save(autoencoder.state_dict(), './supporting_data_files/Autoencoder_512.pt')
    print('Done!')

if __name__ == "__main__":
    main()
