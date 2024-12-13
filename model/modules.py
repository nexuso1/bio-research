import torch
import torch.utils
import math
from dataclasses import dataclass

def keras_init(module):
    """Initialize weights using the Keras defaults."""
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                            torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
        torch.nn.init.uniform_(module.weight, -0.05, 0.05)
    if isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase)):
        for name, parameter in module.named_parameters():
            "weight_ih" in name and torch.nn.init.xavier_uniform_(parameter)
            "weight_hh" in name and torch.nn.init.orthogonal_(parameter)
            "bias" in name and torch.nn.init.zeros_(parameter)
            if "bias" in name and isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
                parameter.data[module.hidden_size:module.hidden_size * 2] = 1
@dataclass
class ConvLayerConfig:
    in_channels : int
    out_channels : int
    kernel_size : int
    num_layers : int
    stride : int

@dataclass
class FusedMBConvConfig(ConvLayerConfig):
    expand : int

class ConvNormActiv1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int, padding : int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = torch.nn.LayerNorm(out_channels)
        self.activ = torch.nn.ReLU()

    def forward(self, inputs : torch.Tensor):
        x = self.conv(inputs)
        x = self.norm(x.moveaxis(-1, 1))
        return self.activ(x.moveaxis(1, -1))
    
class TransposeConvNormActiv1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int, padding : int) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = torch.nn.LayerNorm(out_channels)
        self.activ = torch.nn.ReLU()

    def forward(self, inputs : torch.Tensor):
        x = self.conv(inputs)
        x = self.norm(x.moveaxis(-1, 1))
        return self.activ(x.moveaxis(1, -1))

class Up1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, num_layers : int = 3, kernel_size : int = 3, stride : int = 2,
                 dropout=0):
        super().__init__()
        self.up = TransposeConvNormActiv1D(in_channels, out_channels, kernel_size=3, stride=stride, 
                                           padding=1)
        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(ConvNormActiv1D(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1))
        
        # Needed for Torch to see the list of modules as a part of the graph
        self.layers = torch.nn.ModuleList(self.layers)
        self.dropout= torch.nn.Dropout1d(dropout)
        
    def forward(self, inputs : torch.Tensor, connected : torch.Tensor = None):
        """
        Expects inputs in shape [batch, sequence, channels], 
        """
        x = torch.moveaxis(inputs, -1, 1)
        x = self.up(x)
        if connected is not None:
            connected = torch.moveaxis(connected, -1, 1)
            x = torch.cat([connected, x], 1)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = x + self.layers[i](x) # Residual connection
        return torch.moveaxis(x, 1, -1)

class Down1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, stride=2, dropout=0) -> None:
        super().__init__()
        self.down = ConvNormActiv1D(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=(kernel_size - 1) // 2)
        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(ConvNormActiv1D(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1))
        # Needed for Torch to see the list of modules as a part of the graph
        self.layers = torch.nn.ModuleList(self.layers)
        self.dropout = torch.nn.Dropout1d(dropout)
        
    def forward(self, inputs : torch.Tensor):
        x = torch.moveaxis(inputs, -1, 1)
        x = self.down(x)
        x = self.dropout(x)
        if len(self.layers) > 0:
            x = self.layers[0](x)
            x = self.dropout(x)
        for i in range(1, len(self.layers)):
            x = x + self.layers[i](x) # Residual connection
            x = self.dropout(x)
        return torch.moveaxis(x, 1, -1)
    

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, hidden_size : int, num_layers : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True,
                                  num_layers=num_layers)
        self.outputs = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, inputs : torch.Tensor, lengths):
        # For some reason, after calling pack_padded_sequence, lengths get put back to the gpu 
        # even though they were at cpu before, and the function throws an error. Seems to only happend
        # with torch dynamo. So do not use for now
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(),
                                                        batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[..., :self.hidden_size] + x[..., self.hidden_size:]
        return self.outputs(x)

class Unet1D(torch.nn.Module):
    def __init__(self, layer_configs : list[ConvLayerConfig], out_dim : int) -> None:
        super().__init__()
        self.downs = []
        for in_channels, out_channels, k, n, s in layer_configs:
            self.downs.append(Down1D(in_channels, out_channels, num_layers=n, kernel_size=k, stride=s))
        # Needed for Torch to see the list of modules as a part of the graph
        self.downs = torch.nn.ModuleList(self.downs)

        self.ups = []
        for i in range(len(layer_configs) - 2, -1 , -1):
            connected_channels = layer_configs[i].out_channels
            prev_channels = layer_configs[i+1].out_channels
            self.ups.append(Up1D(prev_channels, connected_channels, num_layers=n, kernel_size=k, stride=s))

        self.ups = torch.nn.ModuleList(self.ups)
        self.final_conv = torch.nn.Conv1d(layer_configs[0].out_channels, out_dim, kernel_size=1, padding=0, stride=1)
        
    def forward(self, inputs):
        down_outs = []
        x = self.downs[0](inputs)
        down_outs.append(x)
        for d in self.downs[1:]:
            x = d(x)
            down_outs.append(x)
        down_outs.reverse()
        for i in range(len(self.ups)):
            x = self.ups[i](x, down_outs[i+1])

        x = self.final_conv(torch.moveaxis(x, -1, 1))
        return torch.moveaxis(x, 1, -1)


class Conv1dModel(torch.nn.Module):
    def __init__(self, layer_configs : list[ConvLayerConfig], pool=True, dropout=0) -> None:
        super().__init__()
        self.downs = []
        for config in layer_configs:
            self.downs.append(Down1D(config.in_channels, config.out_channels, num_layers=config.num_layers,
                                      kernel_size=config.kernel_size, stride=config.stride, dropout=dropout))
        self.downs = torch.nn.ModuleList(self.downs)
        self.pool = pool

    def forward(self, inputs):
        x = self.downs[0](inputs)
        for d in self.downs[1:]:
            x = d(x)

        if self.pool:
            x = x.max(1)[0]

        return x.squeeze()

class FusedMBConv1dModel(torch.nn.Module):
    """
    Based on EfficientNetV2: Smaller Models and Faster Training [https://arxiv.org/abs/2104.00298]
    SE block in the diagram is not used in practice, so it is not used here either
    """
    def __init__(self, layer_configs : list[FusedMBConvConfig], pool=False, dropout=0) -> None:
        super().__init__()
        self.downs = []
        self.activ = torch.nn.ReLU()
        for config in layer_configs:
            self.downs.append(FusedMBConv1D(config.in_channels, config.out_channels, num_layers=config.num_layers,
                                      kernel_size=config.kernel_size, stride=config.stride, dropout=dropout, activ=self.activ,
                                      expand=config.expand))
        self.downs = torch.nn.ModuleList(self.downs)
        self.pool = pool

    def forward(self, inputs):
        x = self.downs[0](inputs)
        for d in self.downs[1:]:
            x = d(x)

        if self.pool:
            x = x.max(1)[0]

        return x.squeeze()
    
class DummyModule(torch.nn.Module):
    def __init__(self, out_shape) -> None:
        super(DummyModule, self).__init__()
        self.out_shape = out_shape

    def forward(self, *args, **kwargs):
        return torch.zeros(size=self.out_shape)

class SinPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len : int):
        super().__init__()

        positions = torch.arange(0, max_len).unsqueeze_(1)
        pe = torch.zeros(max_len, d_model)
        n = 10000
        denominators = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model)) # 10000^(2i/d_model), i is the index of embedding
        pe[:, 0::2] = torch.sin(positions * denominators) # sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(positions * denominators) # cos(pos/10000^(2i/d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` or [batch_size, seq_len, n_head, embedding // n_head]
        """
        if x.dim() == 4:
            return x + self.pe[0, :x.size(1)].view(1, -1, x.size(2), x.size(3))
        
        return x + self.pe[0, :x.size(1)]

    
class ResidualDense(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(ResidualDense, self).__init__()
        self.dense = torch.nn.Linear(input_size, output_size)
        self.res_con = input_size == output_size
        self.activation = activation if activation is not None else torch.nn.Identity()
        
    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.activation(inputs)
        
        if self.res_con:
            x = x + inputs
            
        return x
    
class ResidualMLP(torch.nn.Module):
    def __init__(self, layer_sizes: list[int], input_size,  activation=None, norm=False, dropout=0):
        super(ResidualMLP, self).__init__()
        layer_list = []
        layer_list.append(ResidualDense(input_size, layer_sizes[0], activation))
        for i in range(len(layer_sizes) - 1):
            layer_list.append(ResidualDense(layer_sizes[i], layer_sizes[i + 1]))
        
        self.layers = torch.nn.ModuleList(layer_list)
        self.activation = activation
        if norm is not None:
            norms = []
            norms.append(norm(input_size))
            for i in range(len(layer_list) - 1):
                norms.append(norm(layer_sizes[i]))
            self.norms = torch.nn.ModuleList(norms)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            if len(self.norms) > 0:
                x = self.norms[i](x)

            x = self.layers[i](x)
            x = self.dropout(x)

        x = self.norms[-1](x)
        return self.layers[-1](x)
    
class FusedMBConv1D(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, expand : int, kernel_size : int, stride : int = 1, activ=None, dropout=0):
        super(FusedMBConv1D, self).__init__()
        padding = (kernel_size - 1) // 2
        exp_dim = input_dim * expand
        self.res_con = input_dim == output_dim
        self.expand = torch.nn.Conv1d(input_dim, exp_dim, kernel_size, stride, padding)
        self.activ = activ if activ is not None else torch.nn.Identity()
        self.norm1 = torch.nn.LayerNorm(exp_dim)
        self.project = torch.nn.Conv1d(exp_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = torch.nn.LayerNorm(output_dim)
        self.dropout = torch.nn.Dropout1d(dropout)

    def forward(self, inputs : torch.Tensor):
        """
        Expects input in form of [B, S, CH]
        """
        inputs = inputs.moveaxis(-1, 1)
        x = self.expand(inputs)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.project(x)
        x = self.norm2(x)
        x = self.dropout(x)

        if self.res_con:
            x = x + inputs
        
        return x.moveaxis(1, -1)