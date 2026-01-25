import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import flwr as fl
from captum.attr import LayerGradCam
from captum.attr import visualization as vit

# Note: get_model should be imported or passed if needed, 
# but for now we expect it to be available or passed via model_params

# --- Attribution Helpers ---

def set_inplace(model, mode=False):
    """Recursively sets inplace attribute of all ReLU modules in a model."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = mode

def get_layer_importance(model, target_layer, input_data, target_label, device):
    """Uses Captum LayerGradCam to compute importance per channel."""
    lgc = LayerGradCam(model, target_layer)
    # LayerGradCam expects (input, target)
    attributions = lgc.attribute(input_data, target=target_label)
    importance = torch.mean(attributions, dim=[0, 2, 3])
    importance = torch.abs(importance)
    return importance.cpu().detach().numpy()

# --- FedExp Strategy ---

class FedExpStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_set=None, device='cpu', model_params=None, get_model_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.test_set = test_set
        self.device = device
        self.model_params = model_params # {name, num_classes, input_size, in_channels, dataset_name}
        self.get_model_fn = get_model_fn
        self.vis_dir = os.path.join('results', 'fedexp_vis')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Pre-calculate layer mapping to state_dict indices
        if self.get_model_fn:
            m_args = {k: v for k, v in self.model_params.items() if k != 'dataset_name'}
            ref_model = self.get_model_fn(**m_args)
            self.param_keys = list(ref_model.state_dict().keys())
            self.layer_map = {}
            for i, key in enumerate(self.param_keys):
                if key.endswith('.weight'):
                    base = key[:-7]
                    if base not in self.layer_map: self.layer_map[base] = {'w': None, 'b': None}
                    self.layer_map[base]['w'] = i
                elif key.endswith('.bias'):
                    base = key[:-5]
                    if base not in self.layer_map: self.layer_map[base] = {'w': None, 'b': None}
                    self.layer_map[base]['b'] = i
            
            # Identify possible first FC layer and last conv layer for pathway reinforcement
            self.last_conv_name = None
            for name, module in ref_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.last_conv_name = name
            
            self.first_fc_name = None
            self.first_fc_idx = None
            for i, key in enumerate(self.param_keys):
                if ('.fc' in key or '.classifier' in key or '.linear' in key) and key.endswith('.weight'):
                    self.first_fc_name = key[:-7]
                    self.first_fc_idx = i
                    break

    def aggregate_fit(self, server_round, results, failures):
        # 1. Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and len(results) > 0:
            # 2. Selective Refinement based on Grad-CAM
            params_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            local_weights_list = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
            
            importances_list = []
            for _, res in results:
                if "importance" in res.metrics and res.metrics["importance"]:
                    try:
                        imp = json.loads(res.metrics["importance"])
                        importances_list.append(imp)
                    except:
                        importances_list.append(None)
                else:
                    importances_list.append(None)
            
            valid_imps = [imp for imp in importances_list if imp is not None]
            if len(valid_imps) == len(results) and hasattr(self, 'layer_map'):
                print(f"      [FedExp] Brutal Pathway Refinement across reported convolutional layers...")
                
                shared_layers = list(valid_imps[0].keys())
                for layer_name in shared_layers:
                    if layer_name in self.layer_map:
                        w_idx = self.layer_map[layer_name]['w']
                        b_idx = self.layer_map[layer_name]['b']
                        
                        if w_idx is not None:
                            imp_array = np.array([imp[layer_name] for imp in valid_imps])
                            win_indices = np.argmax(imp_array, axis=0)
                            new_w = np.zeros_like(params_ndarrays[w_idx])
                            num_channels = new_w.shape[0]
                            for c in range(num_channels):
                                winner_idx = win_indices[c]
                                new_w[c] = local_weights_list[winner_idx][w_idx][c]
                            params_ndarrays[w_idx] = new_w
                            
                            if b_idx is not None:
                                new_b = np.zeros_like(params_ndarrays[b_idx])
                                for c in range(num_channels):
                                    winner_idx = win_indices[c]
                                    new_b[c] = local_weights_list[winner_idx][b_idx][c]
                                params_ndarrays[b_idx] = new_b
                
                if self.first_fc_idx is not None and self.last_conv_name in shared_layers:
                    last_imp = np.array([imp[self.last_conv_name] for imp in valid_imps])
                    winners = np.argmax(last_imp, axis=0)
                    fc_w = np.zeros_like(params_ndarrays[self.first_fc_idx])
                    num_channels = len(winners)
                    in_dim = fc_w.shape[1]
                    f_per_c = in_dim // num_channels
                    if f_per_c > 0:
                        for c in range(num_channels):
                            winner_idx = winners[c]
                            start_f = c * f_per_c
                            end_f = (c + 1) * f_per_c
                            fc_w[:, start_f:end_f] = local_weights_list[winner_idx][self.first_fc_idx][:, start_f:end_f]
                        params_ndarrays[self.first_fc_idx] = fc_w
                
                aggregated_parameters = fl.common.ndarrays_to_parameters(params_ndarrays)
                print(f"      [FedExp] Deep feature pathways reinforced.")

            # 3. Shared Visualization
            if self.test_set is not None:
                self.visualize_round(server_round, aggregated_parameters, local_weights_list)

        return aggregated_parameters, aggregated_metrics

    def visualize_round(self, server_round, global_params, local_weights_list):
        if not self.get_model_fn: return
        
        idx = np.random.randint(len(self.test_set))
        img_tensor, label = self.test_set[idx]
        img_input = img_tensor.unsqueeze(0).to(self.device)
        
        def get_net(params):
            m_args = {k: v for k, v in self.model_params.items() if k != 'dataset_name'}
            net = self.get_model_fn(**m_args).to(self.device)
            params_dict = zip(net.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            net.eval()
            set_inplace(net, False)
            return net

        global_net = get_net(fl.common.parameters_to_ndarrays(global_params))
        client_nets = [get_net(w) for w in local_weights_list]
        
        def get_target_layer(net):
            if self.last_conv_name:
                curr = net
                for part in self.last_conv_name.split('.'):
                    curr = getattr(curr, part)
                return curr
            return None

        target_module_global = get_target_layer(global_net)
        if target_module_global is None: return

        def get_hm(net, module):
            lgc = LayerGradCam(net, module)
            attr = lgc.attribute(img_input, target=label)
            attr_map = torch.mean(attr, dim=1).squeeze()
            attr_map = F.relu(attr_map)
            if torch.max(attr_map) > 0:
                attr_map /= torch.max(attr_map)
            attr_np = attr_map.cpu().detach().numpy()
            import cv2
            img_h, img_w = img_input.shape[2], img_input.shape[3]
            attr_np = cv2.resize(attr_np, (img_w, img_h))
            return attr_np

        global_hm = get_hm(global_net, target_module_global)
        client_hms = [get_hm(net, get_target_layer(net)) for net in client_nets]
        
        num_clients = len(client_hms)
        rows = 2
        cols = (num_clients + 2 + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        dataset_name = self.model_params.get('dataset_name', 'cifar10').lower()
        if dataset_name == 'oxfordpet':
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        elif dataset_name == 'mnist':
            mean, std = np.array([0.1307]), np.array([0.3081])
        else:
            mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
            
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        if len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze()

        from matplotlib.colors import LinearSegmentedColormap
        colors = ["red", "yellow", "green"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_rg", colors)
        
        def plot_on(ax, img, hm, title):
            ax.imshow(img)
            if hm is not None:
                ax.imshow(hm, cmap=custom_cmap, alpha=0.4)
            ax.set_title(title, fontsize=14)
            ax.axis('off')

        plot_on(axes[0], img_np, None, "(a) Original Image")
        for i, hm in enumerate(client_hms):
            plot_on(axes[i+1], img_np, hm, f"({chr(98+i)}) Client {i+1}")
            
        plot_on(axes[num_clients + 1], img_np, global_hm, f"({chr(98+num_clients)}) Aggregated Server")

        for j in range(num_clients + 2, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f"round_{server_round}.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"      [FedExp] Visualization saved to {save_path}")
