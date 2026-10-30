import torch
import torch.nn as nn


# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self, activation=nn.ELU(alpha=1.0), dropout=0.25):
        super(EEGNet, self).__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0)),
            nn.Dropout(p=dropout)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )    
        
        
        
    def forward(self, x):
        # x 形狀: [B, 1, 2, 750]
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        
        # 展平成一維 [B, 736]
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x


# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ELU(alpha=1.0), dropout=0.5, C=2, T=750, N=2):
        super(DeepConvNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(C, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        self.block2 = self._conv_block(25, 50, activation, dropout)
        self.block3 = self._conv_block(50, 100, activation, dropout)
        self.block4 = self._conv_block(100, 200, activation, dropout)

        # 用假輸入推算展平後的維度 (C=2, T=750 時為 200*43=8600)
        with torch.no_grad():
            n_features = self._features(torch.zeros(1, 1, C, T)).shape[1]

        self.classify = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=N, bias=True)
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, activation, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 5)),
            nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

    def _features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        # x 形狀: [B, 1, 2, 750]
        x = self._features(x)
        x = self.classify(x)
        return x
