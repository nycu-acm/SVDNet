import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1,
                 activation=torch.nn.functional.relu, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def forward(self,src, tc):
        q = src # (L,B,C)
        k = tc  # (N,B,C)
        src2, att_out = self.self_attn(q, k, value=tc, attn_mask=None,
                              key_padding_mask=None)
        src = self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(1, 2, 0).view(2, 64, 496, -1), att_out


if __name__ == '__main__':
    TFE = TransformerEncoderLayer.cuda()
    print(TFE)
    spatial_features = torch.rand(2,64,486,432).cuda()
    
    spatial_features[:,:,:,216:], _ = TFE(spatial_features[:,:,:,216:].reshape(2, 64, -1).permute(2, 0, 1)
                                          , spatial_features[:,:,:,0:216].reshape(2, 64, -1).permute(2, 0, 1))