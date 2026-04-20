import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class CNN(nn.Module):
    """
    Input:
      x is a list/tuple, we use x_begin = x[0]
      x_begin shape: [B, T, C, H, W]

    Output:
      y shape: [B, H, W, 1]
    """
    def __init__(self, config):
        super().__init__()

        in_ch = config.MODEL.IN_CHANNEL
        if in_ch is None:
            in_ch = 13

        hidden = config.MODEL.TEMPORAL.HIDDEN_DIM
        lstm_layers = getattr(config.MODEL.TEMPORAL, "NUM_LAYERS", 1)
        bidirectional = getattr(config.MODEL.TEMPORAL, "BIDIRECTIONAL", False)
        dropout = getattr(config.MODEL.TEMPORAL, "DROPOUT", 0.0)

        # CNN feature extractor (per-frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # LSTM nhận input theo từng pixel location qua time
        # Mỗi vị trí (h, w) có 1 chuỗi độ dài T, mỗi bước có feature dim = hidden
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden * 2 if bidirectional else hidden

        # Final projection to 1 channel
        self.head = nn.Conv2d(lstm_out_dim, 1, kernel_size=1)

    def forward(self, x):
        x_begin = x[0]  # [B, T, C, H, W]
        B, T, C, H, W = x_begin.shape

        # 1) CNN per frame
        xt = x_begin.view(B * T, C, H, W)          # [B*T, C, H, W]
        feat = self.cnn(xt)                        # [B*T, hidden, H, W]

        hidden = feat.shape[1]
        feat = feat.view(B, T, hidden, H, W)       # [B, T, hidden, H, W]

        # 2) Chuẩn bị cho LSTM
        # Với mỗi pixel (h, w), ta có chuỗi T vector feature dim=hidden
        # [B, T, hidden, H, W] -> [B, H, W, T, hidden]
        feat = feat.permute(0, 3, 4, 1, 2).contiguous()

        # Gộp B*H*W thành batch cho LSTM
        feat = feat.view(B * H * W, T, hidden)     # [B*H*W, T, hidden]

        # 3) LSTM theo time
        lstm_out, (h_n, c_n) = self.lstm(feat)     # lstm_out: [B*H*W, T, lstm_out_dim]

        # Lấy output ở time step cuối
        last_feat = lstm_out[:, -1, :]             # [B*H*W, lstm_out_dim]

        # 4) Khôi phục lại spatial map
        lstm_out_dim = last_feat.shape[-1]
        map_feat = last_feat.view(B, H, W, lstm_out_dim)   # [B, H, W, lstm_out_dim]
        map_feat = map_feat.permute(0, 3, 1, 2).contiguous()  # [B, lstm_out_dim, H, W]

        # 5) Head projection
        y = self.head(map_feat)                    # [B, 1, H, W]

        # Return [B, H, W, 1]
        y = y.permute(0, 2, 3, 1).contiguous()
        return y     
    
