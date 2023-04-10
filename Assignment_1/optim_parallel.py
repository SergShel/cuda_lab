import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import optuna
from optuna.trial import TrialState

import multiprocessing

def get_mnist():
    train_loader =  torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN
    )

    eval_loader =  torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, eval_loader

class Model(nn.Module):
    """
    MLP 
    """
    def __init__(self, input_dim=784, hidden_layers_dims=[128, 64], output_dim=10, activation=nn.ReLU):
        """ Model initalizer """
        super().__init__()

        self.layers = nn.Sequential()
        num_neurons = [input_dim] + hidden_layers_dims + [output_dim]
        for i, in_dims in enumerate(num_neurons[:-2]):
            out_dims = num_neurons[i+1]
            self.layers.add_module(f"{i*2} | Linear_{i:<5}", nn.Linear(in_features=in_dims, out_features=out_dims, bias=True))
            self.layers.add_module(f"{i*2+1} | Activation_{i}", activation())
        self.layers.add_module(f"{len(num_neurons)*2-1} | Linear_{len(num_neurons)-2:<5}", nn.Linear(in_features=num_neurons[-2], out_features=output_dim, bias=True))

        
    def forward(self, x):
        """ Forward pass through the model"""
        assert len(x.shape) == 2, f"ERROR! Shape of input must be 2D (b_size, dim)"
        pred = self.layers(x)
        return pred

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


CLASSES = 10

def define_model(trial):
    # We optimize the number of layers, hidden units in each layer and activation function.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers_dims = []
    for i in range(n_layers):
        max_out_features = 324 // (2 ** i)
        min_out_features = 16 // (2 ** i)
        out_features = trial.suggest_int("n_units_l{}".format(i), min_out_features, max_out_features)
        hidden_layers_dims.append(out_features)
    activation_name = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU"])
    activation = getattr(nn, activation_name)

    model = Model(hidden_layers_dims=hidden_layers_dims, activation=activation)

    return model

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 1024
SHUFFLE_TRAIN = True
N_TRAIN_EXAMPLES = len(train_dataset)
N_VALID_EXAMPLES = len(test_dataset)

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(device)
    criterion = nn.CrossEntropyLoss() 

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, eval_loader = get_mnist()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(eval_loader):
                # Limiting validation data.
                if batch_idx * BATCH_SIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(eval_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def start_study_worker(event):
    event.wait()  # Wait for an event
    study = optuna.load_study(study_name="opt_MLP", storage="mysql://root@localhost/opt_MLP")
    study.optimize(objective, n_trials=200, timeout=1000)


if __name__=="__main__":
    e = multiprocessing.Event() # Create event that will be used for synchronization
    process_list = []
    for i in range(8):
        print(f"Starting worker #{i+1}")
        worker = multiprocessing.Process(target=start_study_worker, args=(e,)) # Create a worker for optuna
        process_list.append(worker)
        worker.start()
    e.set()
    for worker in process_list:
        worker.join() # Wait for all workers to finish
        

    print("All done!")

