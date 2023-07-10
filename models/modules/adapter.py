import torch.nn as nn


class AccentAdapter(nn.Module):
    def __init__(self, out_dim, in_dim, dropout_p=0.1):
        super(AccentAdapter, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(out_dim, eps=1e-5),
            nn.Linear(out_dim, in_dim, bias=True),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):
        return self.sequential(x)
