# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data

myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(93))
        else:
            feats = list(range(40)) + [57, 75]
            # TODO: Using   40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss with L1 regularization '''
        mse_loss = self.criterion(pred, target)
        l1_lambda = 0.005  # Regularization strength
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = mse_loss + l1_lambda * l1_norm
        return loss


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''
    n_epochs = config['n_epochs']  # Maximum number of epochs
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            # print('Saving model (epoch = {:4d}, loss = {:.4f})'
            #     .format(epoch + 1, min_mse))
            # torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

#Validation
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

#Set up Hyper-parameters


device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

import itertools
import random

# Hyperparameter search space
param_grid = {
    'n_epochs': [1000, 2000, 3000],
    'batch_size': [64, 128, 256, 512],
    'optimizer': ['SGD', 'Adam'],
    'optim_hparas':{
        'lr': [0.001,0.0005,0.0001,0.0001],
        'momentum': [0.9,0.7,0.5]
        },
    'early_stop': [50, 100, 200]

}
# config = {
#     'n_epochs': 3000,                # maximum number of epochs
#     'batch_size': 270,               # mini-batch size for dataloader
#     'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
#     'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
#         'lr': 0.001,                 # learning rate of SGD
#         'momentum': 0.9              # momentum for SGD
#     },
#     'early_stop': 200,  
# Number of random samples to take from the hyperparameter space
n_iter = 10

# Function to create a random hyperparameter configuration
import random

param_grid = {
    'n_epochs': [1000, 2000, 3000],
    'batch_size': [64, 128, 256, 512],
    'optimizer': ['SGD', 'Adam'],
    'optim_hparas': {
        'lr': [0.001, 0.0005, 0.0001],
        'momentum': [0.9, 0.7, 0.5]
    },
    'early_stop': [50, 100, 200]
}


def sample_hyperparameters(param_grid):
    random_params = {param: random.choice(values) for param, values in param_grid.items() if param != 'optim_hparas'}
    
    # Handle 'optim_hparas' separately to ensure that 'momentum' is only included for 'SGD'
    optim_hparas = {'lr': random.choice(param_grid['optim_hparas']['lr'])}
    if random_params['optimizer'] == 'SGD':
        optim_hparas['momentum'] = random.choice(param_grid['optim_hparas']['momentum'])
    
    random_params['optim_hparas'] = optim_hparas
    
    return random_params

# Example usage:
random_config = sample_hyperparameters(param_grid)
print(random_config)

# Function to perform the random search
def random_search(param_grid, n_iter, tr_set, dv_set, device):
    best_loss = float('inf')
    best_params = None
    for i in range(n_iter):
        params = sample_hyperparameters(param_grid)
        model = NeuralNet(input_dim=tr_set.dataset.dim).to(device)
        print(f"Training with parameters: {params}")
        mse_loss, _ = train(tr_set, dv_set, model, params, device)
        if mse_loss < best_loss:
            best_loss = mse_loss
            best_params = params
            print(f"New best loss: {best_loss}")
            torch.save(model.state_dict(), 'models/model.pth')
    return best_params

# Include your prep_dataloader function here to create the tr_set and dv_set
# ...

# Conduct random search
tr_set = COVID19Dataset(tr_path, mode='train', target_only=target_only)
dv_set= COVID19Dataset(tr_path, mode='dev', target_only=target_only)
tt_set= COVID19Dataset(tr_path, mode='test', target_only=target_only)

best_hyperparams = random_search(param_grid, n_iter, tr_set, dv_set, device)
print("Best hyperparameters found:", best_hyperparams)

# Now you can train your final model with the best hyperparameters
final_model = NeuralNet(input_dim=tr_set.dataset.dim).to(device)
final_loss, final_loss_record = train(tr_set, dv_set, final_model, best_hyperparams, device)
plot_learning_curve(final_loss_record, title='deep model')

def load_model(model_path, input_dim, device):
    ''' Load the trained model from a specified path '''
    model = NeuralNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

import csv

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p.item()])  # Ensure you convert tensor to a standard Python number with .item()

# Be sure to call the function to save predictions
# Load the best model
model = load_model('models/model.pth', tr_set.dataset.dim, device)

# Predict COVID-19 cases with your model
preds = test(tt_set, model, device)

# Save prediction file to pred.csv
save_pred(preds, 'pred.csv')











# del model
# model = NeuralNet(tr_set.dataset.dim).to(device)
# ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
# model.load_state_dict(ckpt)
# plot_pred(dv_set, model, device)  # Show prediction on the validation set

# def save_pred(preds, file):
#     ''' Save predictions to specified file '''
#     print('Saving results to {}'.format(file))
#     with open(file, 'w') as fp:
#         writer = csv.writer(fp)
#         writer.writerow(['id', 'tested_positive'])
#         for i, p in enumerate(preds):
#             writer.writerow([i, p])

# preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
# save_pred(preds, 'pred.csv')         # save prediction file to pred.csv





# tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
# dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
# tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

# TODO: How to tune these hyper-parameters to improve your model's performance?
# config = {
#     'n_epochs': 3000,                # maximum number of epochs
#     'batch_size': 270,               # mini-batch size for dataloader
#     'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
#     'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
#         'lr': 0.001,                 # learning rate of SGD
#         'momentum': 0.9              # momentum for SGD
#     },
#     'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
#           # your model will be saved here
# }   

# def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
#     ''' Generates a dataset, then is put into a dataloader. '''
#     dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
#     dataloader = DataLoader(
#         dataset, batch_size,
#         shuffle=(mode == 'train'), drop_last=False,
#         num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
#     return dataloader
