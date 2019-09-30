import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_normal_, calculate_gain
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class Laks_net(nn.Module):
    def __init__(self):
        super(Laks_net, self).__init__()
        self.model = Sequential()
        self.model.add("conv_1_1", nn.Conv2d(3, 32, kernel_size=3, padding='same'))
        self.model.add(torch.nn.ReLU())
        self.model.add("conv_1_2", nn.Conv2d(32, 32, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("max_pooling_1", nn.MaxPool2d(kernel_size=2))
        self.model.add("dropout_1", nn.Dropout(0.25))

        self.model.add("conv_2_1", nn.Conv2d(32, 64, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("conv_2_2", nn.Conv2d(64, 64, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("max_pooling_2", nn.MaxPool2d(kernel_size=2))
        self.model.add("dropout_2", nn.Dropout(0.25))

        self.model.add("conv_3_1", nn.Conv2d(64, 128, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("conv_3_2", nn.Conv2d(128, 128, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("max_pooling_3", nn.MaxPool2d(kernel_size=2))
        self.model.add("dropout_3", nn.Dropout(0.25))
        
        self.model.add("conv_4_1", nn.Conv2d(128, 128, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("conv_4_2", nn.Conv2d(128, 128, kernel_size=3, padding='same'))
        self.model.add(nn.ReLU())
        self.model.add("max_pooling_4", nn.MaxPool2d(kernel_size=2))
        self.model.add("dropout_4", nn.Dropout(0.25))
        
        self.classifier = Sequential()
        self.classifier.add('dense1',nn.Linear(in_features=32*32*128, out_features=256, bias=True))
        self.classifier.add(nn.ReLU())
        self.classifier.add('dense2',nn.Linear(in_features=256, out_features=256, bias=True))
        self.classifier.add(nn.ReLU())
        self.classifier.add('dense3',nn.Linear(in_features=256, out_features=2, bias=True))

        self.model.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, x):
        feature = self.model(x)
        feature.view(feature.size(0), -1)
        logits = self.classifier(feature)
        return logits.reshape(-1,2)
