
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import DynamicCNN
from utils import get_device, count_parameters
import numpy as np
from tqdm import tqdm
from dataloaders import cifar_test_loader, cifar_train_loader,custom_train_loader
from collections import OrderedDict
import copy


EPOCHS = 1000
LEARNING_RATE = 2e-3
UPGRADE_AMT = 4
DROPOUT = 0.1
image_size = 32
channels_list = [3, 16, 16, 16]
n_classes = 10
num_clients = 2
num_selected = 2
num_rounds = 29
batch_size = 64
L1_REG = 1e-5
expansion_threshold = 2


device = get_device()

traindata_split = torch.utils.data.random_split(cifar_train_loader.dataset,
                                                [int(len(cifar_train_loader.dataset) / num_clients) for _ in
                                                 range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
test_loader = DataLoader(cifar_test_loader.dataset, batch_size=batch_size, shuffle=True)
retrain_loader = DataLoader(custom_train_loader.dataset, batch_size=batch_size, shuffle=True)
simulated_device_capabilities = [
    {'compute_power': 4.0, 'memory': 4.0, 'bandwidth': 128, 'max_expansions': 2},
    {'compute_power': 24.0, 'memory': 24.0, 'bandwidth': 768, 'max_expansions': 5}
]

# device_selection_mode = "random"
device_selection_mode = "manual"
manual_device_assignment = [0, 1]

def measure_device_capability(device_index):
    if device_index < 0 or device_index >= len(simulated_device_capabilities):
        raise IndexError("Device index out of range")
    return simulated_device_capabilities[device_index]



class IdentityConvLayer(nn.Module):
    """
    An identity conv layer with weights initialized to Identity, with a bit of gausian noise added.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(channels, channels,
                         kernel_size=3, padding="same", bias=False)

        # Creating an identity matrix with added noise
        identity_matrix = torch.eye(channels).view(channels, channels, 1, 1)
        noise = torch.randn(identity_matrix.shape) * NOISE_COEFF
        identity_matrix_with_noise = identity_matrix + noise
        with torch.no_grad():
            conv.weight.copy_(identity_matrix_with_noise)
        self.conv = nn.Sequential(conv,
                                  nn.BatchNorm2d(channels),
                                  nn.LeakyReLU(0.2)).to(device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)



def obs_prune_and_compensate1(model, gradients, H, device):
    gradients = gradients.to(device)
    H = H.to(device)
    weights = []
    for param in model.parameters():
        weights.append(param.view(-1))

    weights = torch.cat(weights).to(device)

    delta_L = (gradients ** 2) / (H.diag() + 1e-6)
    min_index = torch.argmin(delta_L)
    weights[min_index] = 0

    new_weights = weights - H.inverse() @ gradients
    start = 0
    for param in model.parameters():
        param_length = len(param.view(-1))
        param.data = new_weights[start:start+param_length].view(param.size())
        start += param_length

    return model
def obs_prune_and_compensate2(model, gradients, H, device):
    gradients = gradients.to(device)
    H = H.to(device)
    weights = []
    for param in model.parameters():
        weights.append(param.view(-1))

    weights = torch.cat(weights).to(device)

    reg_strength = 1e-5
    H += reg_strength * torch.eye(len(weights)).to(device)


    new_weights = weights - torch.linalg.solve(H, gradients)
    start = 0
    for param in model.parameters():
        param_length = len(param.view(-1))
        param.data = new_weights[start:start+param_length].view(param.size())
        start += param_length

    return model

def compute_gradients_and_hessian(model, train_loader, device):
    # model.to(device)
    model.zero_grad()
    gradients = []
    weights = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
            weights.append(param.view(-1))

    if len(gradients) == 0:
        return None, None

    gradients = torch.cat(gradients).to(device)
    weights = torch.cat(weights).to(device)


    H_diag = torch.abs(gradients) + 1e-6
    H = torch.diag(H_diag)

    return gradients, weights, H



def prune_and_compensate_layer(model, target_layer_type, gradients, weights, H, device):
    model.to(device)

    layer_indices = []
    pruned_layer_info = {}
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, target_layer_type):
            layer_indices.append(i)
            pruned_layer_info[name] = {
                'index': i,
                'weights': model.state_dict()[name + '.weight'].clone(),
                'bias': model.state_dict()[name + '.bias'].clone() if module.bias is not None else None
            }

    if not layer_indices:
        print("No target layer found.")
        return model, pruned_layer_info


    new_weights = weights.clone()


    for idx in layer_indices:
        start = sum(p.numel() for p in list(model.parameters())[:idx])
        end = start + list(model.parameters())[idx].numel()

        new_weights[start:end] = 0

        H_inv_diag = 1.0 / (H.diag() + 1e-6)
        compensation = H_inv_diag[start:end] * gradients[start:end]
        new_weights[start:end] -= compensation


    start = 0
    for param in model.parameters():
        param_length = len(param.view(-1))
        param.data = new_weights[start:start + param_length].view(param.size())
        start += param_length

    return model, pruned_layer_info
def add_and_compensate_layer(model, target_layer_type, pruned_layer_info, gradients, H, device):
    model.to(device)

    for layer_name, layer_data in pruned_layer_info.items():

        restored_layer = target_layer_type(layer_data['weights'].shape[0])
        restored_layer.to(device)


        with torch.no_grad():
            restored_layer.conv[0].weight.copy_(layer_data['weights'])
            if layer_data['bias'] is not None:
                restored_layer.conv[0].bias.copy_(layer_data['bias'])


        model.add_module(layer_name, restored_layer)


        start = sum(p.numel() for p in list(model.parameters())[:layer_data['index']])
        end = start + restored_layer.conv[0].weight.numel()

        H_inv_diag = 1.0 / (H.diag() + 1e-6)
        compensation = H_inv_diag[start:end] * gradients[start:end]
        with torch.no_grad():
            restored_layer.conv[0].weight.view(-1).add_(compensation)

    return model



def client_update(client_model, optimizer, train_loader, device_index, epochs=1, val_split=0.1):
    client_model.train()
    train_len = int(len(train_loader.dataset) * (1 - val_split))
    val_len = len(train_loader.dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

    current_param_count = sum(p.numel() for p in client_model.parameters() if p.requires_grad)
    prev_val_loss = float('inf')

    device_capability = measure_device_capability(device_index)
    expanded = False
    pruned_layer_info = {}

    for epoch in range(epochs):
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(client_model.device), labels.to(client_model.device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            l1_reg = torch.tensor(0.).to(client_model.device)
            for param in client_model.parameters():
                l1_reg += torch.norm(param, 1)
            loss = criterion(outputs, labels)
            loss_ = loss + L1_REG * l1_reg
            loss_.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        print(f'Client Training - Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%')

        expanded = client_model.expand_if_necessary(train_loader, expansion_threshold, criterion, UPGRADE_AMT, device_capability)
        if expanded:
            print(f'Expanded model at client during epoch {epoch + 1}')
            print("Expanded model structure:")
            print(client_model)
            current_param_count = sum(p.numel() for p in client_model.parameters() if p.requires_grad)

        val_loss = 0.0
        client_model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(client_model.device), labels.to(client_model.device)
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        param_increase = sum(p.numel() for p in client_model.parameters() if p.requires_grad) - current_param_count
        client_model.adjust_lambda_n(val_loss, param_increase, prev_val_loss)
        prev_val_loss = val_loss

    if expanded:

        gradients, weights, H = compute_gradients_and_hessian(client_model, train_loader, client_model.device)
        print("Gradients before pruning:")
        print(gradients)


        print("Evaluating model performance before pruning and compensation...")
        client_model.eval()
        val_loss_before, val_correct_before, val_total_before = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(client_model.device), labels.to(client_model.device)
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                val_loss_before += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_before += labels.size(0)
                val_correct_before += (predicted == labels).sum().item()

        val_loss_before /= len(val_loader)
        val_accuracy_before = 100 * val_correct_before / val_total_before
        print(f'Validation before pruning and compensation - Loss: {val_loss_before:.4f}, Accuracy: {val_accuracy_before:.2f}%')


        client_model, pruned_layer_info = prune_and_compensate_layer(client_model, IdentityConvLayer, gradients, weights, H, client_model.device)


        new_gradients, _, _ = compute_gradients_and_hessian(client_model, train_loader, client_model.device)
        print("Gradients after compensation:")
        print(new_gradients)


        print("Evaluating model performance after pruning and compensation...")
        client_model.eval()
        val_loss_after, val_correct_after, val_total_after = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(client_model.device), labels.to(client_model.device)
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                val_loss_after += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_after += labels.size(0)
                val_correct_after += (predicted == labels).sum().item()

        val_loss_after /= len(val_loader)
        val_accuracy_after = 100 * val_correct_after / val_total_after
        print(f'Validation after pruning and compensation - Loss: {val_loss_after:.4f}, Accuracy: {val_accuracy_after:.2f}%')

    return client_model.state_dict(), pruned_layer_info if expanded else None



def server_aggregatee(global_model, client_models, pruned_layer_infos):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

    for i, pruned_layer_info in enumerate(pruned_layer_infos):
        if pruned_layer_info:
            gradients, _, H = compute_gradients_and_hessian(global_model, train_loader[i], global_model.device)
            client_models[i] = add_and_compensate_layer(client_models[i], IdentityConvLayer, pruned_layer_info, gradients, H, global_model.device)

def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

    fine_tune_global_model(global_model, retrain_loader)


def fine_tune_global_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    device = get_device()
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}')


import gym

class FineTuneEnv(gym.Env):
    def __init__(self, model, data_loader, initial_lr=0.001):
        super(FineTuneEnv, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.initial_lr = initial_lr
        self.device = get_device()

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.current_step = 0
        self.max_steps=10

    def step(self, action):
        if action == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
        elif action == 2:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.1

        running_loss = 0.0
        self.model.train()
        for images, labels in self.data_loader:
            images, labels = images.to(self.device), labels.to(self.device)


            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()


        reward = -running_loss / len(self.data_loader)

        current_lr = self.optimizer.param_groups[0]['lr']
        state = np.array([current_lr], dtype=np.float32)

        self.current_step += 1
        done = (running_loss < 0.09) or (self.current_step >= self.max_steps)
        return state, reward, done, {}

    def reset(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        return np.array([self.initial_lr], dtype=np.float32)


def test(global_model, test_loader):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

global_model = DynamicCNN(channels_list=channels_list,
                          n_classes=n_classes,
                          image_size=image_size,
                          pooling_stride=2,
                          dropout=DROPOUT).to(device)

initial_state = global_model.state_dict()

client_models = [DynamicCNN(channels_list=channels_list,
                            n_classes=n_classes,
                            image_size=image_size,
                            pooling_stride=2,
                            dropout=DROPOUT).to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(initial_state)

opt = [torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in client_models]

criterion = nn.CrossEntropyLoss()

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "num_parameters": [],
    "learning_rate": []
}


print("Initialized client models:")
for i,model in enumerate(client_models):
    print(f"Client {i} model: {model}")
from stable_baselines3 import DQN
num_rounds=100
pruned_layer_infos = []
for round in range(num_rounds):
    client_indices = np.random.permutation(num_clients)[:num_selected]
    client_states = []
    pruned_layer_infos.clear()

    for i in tqdm(range(num_selected)):
        device_index = client_indices[i] if device_selection_mode == "random" else manual_device_assignment[i]
        state, pruned_layer_info = client_update(client_models[i], opt[i], train_loader[client_indices[i]], device_index, epochs=100)
        client_states.append(state)
        pruned_layer_infos.append(pruned_layer_info)

    server_aggregatee(global_model, client_states, pruned_layer_infos)

    print("Global model parameters after aggregation:")
    for name, param in global_model.named_parameters():
        print(f"{name}:{param.shape}")

    test_loss, acc = test(global_model, test_loader)
    history['val_loss'].append(test_loss)
    history['val_accuracy'].append(acc)

    print(f'{round + 1}-th round | Test Loss: {test_loss:.4f} | Test Accuracy: {acc:.4f}')

    num_params = count_parameters(global_model)
    history['num_parameters'].append(num_params)
    history['learning_rate'].append(LEARNING_RATE)

print('Training completed.')

final_test_loss, final_acc = test(global_model, test_loader)
print(f'Final Test Loss: {final_test_loss:.4f} | Final Test Accuracy: {final_acc:.4f}')
