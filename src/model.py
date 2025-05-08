import torch.nn as nn
from torchvision import transforms

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.fc  = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        # x: (T, B, nIn)
        r, _ = self.rnn(x)           # (T, B, 2*H)
        T, B, H2 = r.size()
        r = r.view(T * B, H2)        # (T*B, 2*H)
        out = self.fc(r)             # (T*B, nOut)
        return out.view(T, B, -1)    # (T, B, nOut)

class CRNN(nn.Module):
    def __init__(self,
                 num_classes,
                 imgH: int = 32,
                 nc: int = 1,
                 nh: int = 512,
                 dropout_p: float = 0.3):
        """
        nh: hidden size of RNN layers
        dropout_p: drop probability between RNN layers
        """
        super().__init__()
        assert imgH % 16 == 0, "imgH must be multiple of 16"

        # 1) Convolutional backbone (same as before)
        self.cnn = nn.Sequential(
            nn.Conv2d(nc,  64,  3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128,  3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128,256,  3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256,256,  3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256,512,  3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,  3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512, nh, (imgH//16), 1, 0), nn.ReLU(True)
        )

        # 2) Three-layer BiLSTM with dropout in between
        self.rnn1    = BidirectionalLSTM(nh,  nh,  nh)
        self.drop1   = nn.Dropout(dropout_p)
        self.rnn2    = BidirectionalLSTM(nh,  nh,  nh)
        self.drop2   = nn.Dropout(dropout_p)
        self.rnn3    = BidirectionalLSTM(nh,  nh,  num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        """
        x: (B, nc, imgH, W)
        returns logits: (T, B, C) where T ≈ W/4 and C=num_classes+1
        """
        # → CNN
        c = self.cnn(x)                  # (B, nh, 1, W')
        b, nh, _, w = c.size()
        c = c.squeeze(2).permute(2, 0, 1) # (W', B, nh) = (T, B, nh)

        # → RNN stack
        h = self.rnn1(c)     # (T, B, nh)
        h = self.drop1(h)
        h = self.rnn2(h)     # (T, B, nh)
        h = self.drop2(h)
        o = self.rnn3(h)     # (T, B, C)

        return o
