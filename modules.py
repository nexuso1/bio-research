import torch
import torch.utils
import torchvision
from collections import namedtuple

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

class ConvNormActiv1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int, padding : int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.activ = torch.nn.ReLU()

    def forward(self, inputs : torch.Tensor):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.activ(x)
    
class TransposeConvNormActiv1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int, padding : int) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.activ = torch.nn.ReLU()

    def forward(self, inputs : torch.Tensor):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.activ(x)

class Up1D(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, num_layers : int = 3, kernel_size : int = 3, stride : int = 2):
        super().__init__()
        self.up = TransposeConvNormActiv1D(in_channels, out_channels, kernel_size=2, stride=stride, 
                                           padding=1)
        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(ConvNormActiv1D(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1))
        
        # Needed for Torch to see the list of modules as a part of the graph
        self.layers = torch.nn.ModuleList(self.layers)
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
    def __init__(self, in_channels, out_channels, num_layers=3, kernel_size=3, stride=2) -> None:
        super().__init__()
        self.down = ConvNormActiv1D(in_channels, out_channels, kernel_size=2, stride=stride, 
                                    padding=1)
        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(ConvNormActiv1D(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1))
        # Needed for Torch to see the list of modules as a part of the graph
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, inputs : torch.Tensor):
        x = torch.moveaxis(inputs, -1, 1)
        x = self.down(x)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = x + self.layers[i](x) # Residual connection
        return torch.moveaxis(x, 1, -1)
    

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, hidden_size : int, num_layers : int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True,
                                  num_layers=num_layers)
        self.outputs = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, inputs : torch.Tensor, lengths : torch.Tensor):
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[..., :self.hidden_size] + x[..., self.hidden_size:]
        return self.outputs(x)

class Unet1D(torch.nn.Module):
    LayerConfig = namedtuple('LayerConfig', 'in_channels, out_channels, kernel_size, num_layers, stride')
    def __init__(self, layer_configs : list[LayerConfig], out_dim : int) -> None:
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


        