
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import matplotlib.pyplot as plt
import os
import csv
import shutil
import argparse
from sklearn.metrics import f1_score

# --- 1. Data Preparation ---
def get_dataset(name, img_size):
    if name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
    
    elif name.lower() == 'stl10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set.labels)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
        
    return train_set, test_set, targets

def partition_data(dataset, targets, num_clients, distribution, alpha, num_classes):
    indices = [[] for _ in range(num_clients)]
    
    if distribution.lower() == 'iid':
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = np.array_split(all_indices, num_clients)
        # Convert to list of lists for consistency
        indices = [idx.tolist() for idx in indices]
        
    else: # non-iid (Dirichlet)
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            
            # Dirichlet split
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                indices[i].extend(idx_splits[i])
                
        for i in range(num_clients):
            np.random.shuffle(indices[i])
            
    return indices

# --- 2. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute FC input size dynamically
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        dummy_out = self._forward_features(dummy_input)
        self.fc_input_dim = dummy_out.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Grad-CAM Utilities ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] 
        
    def generate_importance(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(len(class_idx)):
            one_hot[i][class_idx[i]] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3]) 
        pooled_gradients = torch.nan_to_num(pooled_gradients)
        channel_importance = torch.mean(torch.abs(pooled_gradients), dim=0) 
        return channel_importance

    def generate_heatmap(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        pooled_gradients = torch.mean(gradients, dim=[2, 3]) 
        
        for i in range(64):
            activations[:, i, :, :] *= pooled_gradients[:, i].view(-1, 1, 1)
            
        heatmap = torch.sum(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().detach().numpy()

# --- 4. FL Logic & Visualization ---
def train_client(model, loader, epochs, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def compute_client_importance(model, loader):
    model.eval()
    grad_cam = GradCAM(model, model.conv3)
    total_importance = torch.zeros(64).to(args.device)
    count = 0
    batch_count = 0
    for data, target in loader:
        if batch_count > 2: break
        data, target = data.to(args.device), target.to(args.device)
        out = model(data)
        preds = out.argmax(dim=1)
        mask = preds == target
        if mask.sum() > 0:
            clean_data = data[mask]
            imp = grad_cam.generate_importance(clean_data)
            total_importance += imp
            count += 1
        batch_count += 1
        
    if count > 0:
        return total_importance / count
    else:
        return torch.ones(64).to(args.device)

def save_visualization(model, loader, client_id, round_id, img_size, dataset_name, vis_dir):
    model.eval()
    grad_cam = GradCAM(model, model.conv3)
    
    found = False
    for data, target in loader:
        data, target = data.to(args.device), target.to(args.device)
        out = model(data)
        preds = out.argmax(dim=1)
        mask = preds == target
        if mask.any():
            idx = torch.where(mask)[0][0]
            img = data[idx].unsqueeze(0) 
            lbl = target[idx].item()
            pred = preds[idx].item()
            
            heatmap = grad_cam.generate_heatmap(img)
            
            img_vis = img.squeeze().cpu().permute(1, 2, 0).numpy()
            img_vis = img_vis * 0.5 + 0.5
            img_vis = np.clip(img_vis, 0, 1)
            
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_vis)
            plt.title(f"Original (True: {lbl})")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title("Grad-CAM")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(img_vis)
            plt.imshow(heatmap, cmap='jet', alpha=0.5, extent=(0, img_size, img_size, 0))
            plt.title(f"Overlay")
            plt.axis('off')
            
            plt.suptitle(f"Client {client_id} - Round {round_id} - {dataset_name}")
            plt.savefig(f"{vis_dir}/round_{round_id}_client_{client_id}.png")
            plt.close()
            found = True
            break
        if found: break

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate weighted F1 score
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return f1

def aggregate_weights(global_model, local_weights_list, method='fedavg', importances=None, round_id=0, num_clients=4):
    new_state = copy.deepcopy(global_model.state_dict())
    
    if method == 'fedavg':
        for key in new_state.keys():
            new_state[key] = torch.stack([w[key] for w in local_weights_list]).mean(dim=0)
            
    elif method == 'fedgradcam':
        # Baseline average for all layers
        for key in new_state.keys():
            new_state[key] = torch.stack([w[key] for w in local_weights_list]).mean(dim=0)
            
        # Refinement for Conv3
        if importances is not None:
            imp_tensor = torch.stack(importances) 
            voting_weights = imp_tensor / (imp_tensor.sum(dim=0, keepdim=True) + 1e-6)
            
            with open('channel_importance_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                for c in range(64):
                    row = [round_id, c] + voting_weights[:, c].tolist()
                    writer.writerow(row)
            
            local_conv3_w = torch.stack([w['conv3.weight'] for w in local_weights_list]) 
            local_conv3_b = torch.stack([w['conv3.bias'] for w in local_weights_list])   
            
            w_broadcast = voting_weights.view(num_clients, 64, 1, 1, 1)
            
            new_conv3_w = (local_conv3_w * w_broadcast).sum(dim=0)
            new_conv3_b = (local_conv3_b * voting_weights).sum(dim=0)
            
            new_state['conv3.weight'] = new_conv3_w
            new_state['conv3.bias'] = new_conv3_b
            
    return new_state

# --- 5. Simulation Runner ---
def run_simulation(mode='fedavg', train_data=None, test_loader=None, client_indices=None, args=None):
    print(f"\n=== Starting Simulation: {mode} ===")
    
    if mode == 'fedgradcam':
        with open('channel_importance_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Round', 'Channel_ID'] + [f'Client_{i}' for i in range(args.num_clients)]
            writer.writerow(header)
    
    client_loaders = []
    for idxs in client_indices:
        ds = Subset(train_data, idxs)
        client_loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True))
        
    global_model = SimpleCNN(num_classes=args.num_classes, input_size=args.img_size).to(args.device)
    acc_history = []
    time_history = []
    start_time = time.time()
    
    for r in range(args.rounds):
        local_weights = []
        client_importances = []
        
        print(f"[{mode}] Round {r+1}/{args.rounds}")
        
        for i in range(args.num_clients):
            local_model = copy.deepcopy(global_model)
            w_local = train_client(local_model, client_loaders[i], args.epochs, args.lr)
            local_weights.append(w_local)
            
            if mode == 'fedgradcam':
                local_model.load_state_dict(w_local)
                try:
                    imp = compute_client_importance(local_model, client_loaders[i])
                    client_importances.append(imp)
                    save_visualization(local_model, client_loaders[i], i, r+1, args.img_size, args.dataset, args.vis_dir)
                except Exception as e:
                    print(f"Error in GradCAM for client {i}: {e}")
                    client_importances.append(torch.ones(64).to(args.device))
        
        new_weights = aggregate_weights(global_model, local_weights, method=mode, importances=client_importances, round_id=r+1, num_clients=args.num_clients)
        global_model.load_state_dict(new_weights)
        
        # Metric
        f1_val = evaluate(global_model, test_loader)
        acc_history.append(f1_val)
        
        elapsed = time.time() - start_time
        time_history.append(elapsed)
        
        print(f"[{mode}] Round {r+1} F1: {f1_val:.4f} | Time: {elapsed:.1f}s")
        
    return acc_history, time_history

# --- 6. Main & Argument Parsing ---
def main():
    parser = argparse.ArgumentParser(description='Federated Learning with Grad-CAM aggregation')
    parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'cifar10'], help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--distribution', type=str, default='non-iid', choices=['iid', 'non-iid'], help='Data distribution')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-iid')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--epochs', type=int, default=2, help='Local epochs per client')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory for visualizations')
    
    global args
    args = parser.parse_args()
    args.num_classes = 10 
    args.img_size = 128 if args.dataset == 'stl10' else 32

    if os.path.exists(args.vis_dir):
        shutil.rmtree(args.vis_dir)
    os.makedirs(args.vis_dir)

    print(f"Configuration: {args}")

    print(f"Loading {args.dataset} dataset...")
    train_full, test_full, targets = get_dataset(args.dataset, args.img_size)
    client_indices = partition_data(train_full, targets, args.num_clients, args.distribution, args.alpha, args.num_classes)
    test_loader = DataLoader(test_full, batch_size=args.batch_size)
    
    # Run FedAvg
    acc_avg, time_avg = run_simulation('fedavg', train_full, test_loader, client_indices, args)
    
    # Run FedGradCAM
    acc_grad, time_grad = run_simulation('fedgradcam', train_full, test_loader, client_indices, args)
    
    # Final Comparison Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.rounds+1), acc_avg, label='FedAvg', marker='o')
    plt.plot(range(1, args.rounds+1), acc_grad, label='FedGradCAM', marker='s')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Comparison ({args.dataset}, {args.distribution})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.rounds+1), time_avg, label='FedAvg', marker='o')
    plt.plot(range(1, args.rounds+1), time_grad, label='FedGradCAM', marker='s')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Time (s)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True)
    
    save_path = f'simulation_comparison_{args.dataset}_{args.distribution}.png'
    plt.savefig(save_path)
    print(f"\nFinal Results saved to {save_path}")

if __name__ == "__main__":
    main()
