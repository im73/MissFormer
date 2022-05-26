import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", split = False):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.split = split
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

    def forward(self, x, cross, x_mask=None, cross_mask=None, x_p = None, cross_p = None):
        if self.split:
            
            x, x_p, attn = self.self_attention(
                x_p, x, x_p, x, x_p, x,
                attn_mask=x_mask
            )
            x = x + self.dropout(x)
            x_p = x_p + self.dropout(x_p)

            x = self.norm1(x)
            x_p = self.norm4(x_p)

            x, x_p, attn = self.cross_attention(
                x_p, x, cross_p, cross, cross_p, cross,
                attn_mask=x_mask
            )
            x = x + self.dropout(x)
            x_p = x_p + self.dropout(x_p)

            y_x = x = self.norm2(x)
            y_p = x_p = self.norm5(x_p)

            y_x = self.dropout(self.activation(self.conv1(y_x.transpose(-1,1))))
            y_x = self.dropout(self.conv2(y_x).transpose(-1,1))

            y_p = self.dropout(self.activation(self.conv1(y_p.transpose(-1,1))))
            y_p = self.dropout(self.conv2(y_p).transpose(-1,1))

            return self.norm3(x+y_x), self.norm4(x_p+y_p)

        else:
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask
            )[0])
            x = self.norm1(x)

            x = x + self.dropout(self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )[0])

            y = x = self.norm2(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
            y = self.dropout(self.conv2(y).transpose(-1,1))

            return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, split = False):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.split = split

    def forward(self, x, cross, x_mask=None, cross_mask=None, x_p=None, cross_p= None):
        if self.split:
            for layer in self.layers:
                x, x_p = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, x_p = x_p, cross_p = cross_p)

            if self.norm is not None:
                x = self.norm(x + x_p)
        else:
            for layer in self.layers:
                x= layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            if self.norm is not None:
                x = self.norm(x)

        return x


class missDecoderLayer(nn.Module):
    def __init__(self, MaskedGatedBlock, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(missDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.MaskedGatedBlock = MaskedGatedBlock
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.activation2 = F.relu if activation == "relu" else F.gelu
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

    def forward(self, x_v, cross_v, x_mask=None, cross_mask=None, x_p = None, cross_p = None):
        x_val, x_pos, x_mask = self.MaskedGatedBlock((x_v, x_p, x_mask))
        x_val = self.norm1(x_val)
        x_pos = self.norm4(x_pos)
        
        x_val, x_pos, attn = self.cross_attention(
                x_pos, x_val, cross_p, cross_v, cross_p, cross_v,
                attn_mask=x_mask
            )
        x_val = x_val + self.dropout(x_val)
        x_pos = x_pos + self.dropout(x_pos)



        y = x = self.norm2(x_val)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x_val = self.norm3(x+y)

        y = x = self.norm5(x_pos)
        y = self.dropout(self.activation2(self.conv3(y.transpose(-1,1))))
        y = self.dropout(self.conv4(y).transpose(-1,1))
        x_pos = self.norm6(x+y)


        return x_val, x_pos, x_mask

class missDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, split = False):
        super(missDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.split = split

    def forward(self, x, cross, x_mask=None, cross_mask=None, x_p=None, cross_p= None):
        if self.split:
            for layer in self.layers:
                x, x_p, x_mask = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, x_p = x_p, cross_p = cross_p)

            if self.norm is not None:
                x = self.norm(x + x_p)
        else:
            for layer in self.layers:
                x= layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            if self.norm is not None:
                x = self.norm(x)

        return x