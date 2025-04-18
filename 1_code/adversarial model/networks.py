import torch
import torch.nn as nn

class EnsembleNetwork(torch.nn.Module):
    def __init__(self,encoder,decoder, avg_pool):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.avg_pool = avg_pool
        
    def enc_feat(self,input):
        features = self.encoder(input)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self,input):
        features = self.encoder(input)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class DistillerModel(torch.nn.Module):
    def __init__(self, encoder, in_features):
        super().__init__()
        self.in_features = in_features
        self.encoder = encoder
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.decoder = nn.Sequential(
            nn.Linear(self.in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Linear(in_features, 1),
            # nn.Tanh()
        )

    def forward(self, input):
        features = self.encoder(input)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ClassifierModel(torch.nn.Module):
    def __init__(self, encoder, in_features):
        super().__init__()
        self.in_features = in_features
        self.encoder = encoder
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.decoder = nn.Sequential(
            nn.Linear(self.in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
            # nn.Linear(in_features, 2)
        )

    def enc_feat(self,input):
        features = self.encoder(input)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input):
        features = self.encoder(input)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x