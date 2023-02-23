import torch.optim as optim

class custom_optimizer:
    def __init__(self, _optimizer, _layerwise_decay_rate):
        self.optimizer = _optimizer
        self.layerwise_decay_rate = _layerwise_decay_rate
        self.total_steps = 0

    def step(self, *args, **kwargs):

        #for i, group in enumerate(self.optimizer.param_groups):
        #    group['lr'] *= 0.99
        #    print(group['lr'])
        self.optimizer.step(*args, **kwargs)
        self.total_steps += 1
        
    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)