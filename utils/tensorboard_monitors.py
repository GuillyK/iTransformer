import torch
import random

class WeightChangeMonitor:
    def __init__(self, model, top_k=5):
        self.previous_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.top_k = top_k

    def get_layers_with_most_change(self, model):
        weight_changes = {}
        for name, param in model.named_parameters():
            change = torch.norm(
                param.data - self.previous_weights[name]
            ).item()
            weight_changes[name] = change
            self.previous_weights[name] = param.data.clone()

        return sorted(weight_changes, key=weight_changes.get, reverse=True)[
            : self.top_k
        ]

    def get_layers_with_highest_gradients(self, model, top_k=5):
        grad_magnitudes = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_magnitudes[name] = torch.norm(param.grad).item()

        return sorted(grad_magnitudes, key=grad_magnitudes.get, reverse=True)[
            :top_k
        ]

    def get_random_layers(self, model, k=5):
        all_layers = list(model.named_parameters())
        return random.sample(all_layers, min(k, len(all_layers)))

    def get_layers_to_monitor(
        self, model, weight_monitor, top_k=5, random_k=2
    ):
        grad_layers = weight_monitor.get_layers_with_highest_gradients(model, top_k)
        change_layers = weight_monitor.get_layers_with_most_change(model)
        random_layers = [
            name for name, _ in weight_monitor.get_random_layers(model, random_k)
        ]

        return list(set(grad_layers + change_layers + random_layers))




