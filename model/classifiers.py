from torch.nn.modules import Module
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from modules import RNNClassifier
from dataclasses import dataclass, field
from modules import Conv1dModel, ConvLayerConfig

import torch

@dataclass
class RNNTokenClassiferConfig(TokenClassifierConfig):
    hidden_size : int = 256
    n_layers : int = 2
    sr_dim : int = None
    cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
                ConvLayerConfig(1280, 64, 7, 2, 2),
                ConvLayerConfig(64, 128, 5, 2, 2),
                ConvLayerConfig(128, 256, 3, 2, 2),
                ConvLayerConfig(256, 384, 3, 1, 2)
            ])

@dataclass
class EncoderClassifierConfig(TokenClassifierConfig):
    hidden_size : int = 256
    n_heads : int = 8
    n_layers : int = 1
    sr_dim : int = 128
    sr_cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 640, 7, 2, 2),
            ConvLayerConfig(640, 320, 5, 2, 2),
            ConvLayerConfig(320, 160, 3, 2, 2),
            ConvLayerConfig(160, 128, 3, 1, 2)
        ])
    
    res_cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 640, 7, 2, 1),
            ConvLayerConfig(640, 320, 5, 2, 1),
            ConvLayerConfig(320, 160, 3, 2, 1),
            ConvLayerConfig(160, 128, 3, 1, 1)
        ])

@dataclass
class SelectiveFinetuningClassifierConfig(TokenClassifierConfig):
    unfreeze_indices : list[int] = field(default_factory= lambda : [-1])

class LinearClassifier(TokenClassifier):
    def __init__(self, config: TokenClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)
        self.init_weights(self.classifier)

class EncoderClassifier(TokenClassifier):
    def __init__(self, config: TokenClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        enc_layer = torch.nn.TransformerEncoderLayer(config.sr_dim * 2, nhead=config.n_heads,
                                                    dim_feedforward=config.hidden_size, dropout=0,
                                                    activation='relu')
        self.encoder = torch.nn.TransformerEncoder(enc_layer, config.n_layers)
        self.sr_cnn = Conv1dModel(config.sr_cnn_layers, config.cnn_layers[-1].out_channels)
        self.res_cnn = Conv1dModel(config.res_cnn_layers, )
        self.seq_rep = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.BatchNorm1d(config.cnn_layers[-1].out_channels),
                torch.nn.Linear(config.cnn_layers[-1].out_channels, config.sr_dim),
                torch.nn.ReLU()
            )
        
        self.res_rep = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.BatchNorm1d(config.cnn_layers[-1].out_channels),
                torch.nn.Linear(config.cnn_layers[-1].out_channels, config.sr_dim),
                torch.nn.ReLU()
            )
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.n_labels),
        )

        self.init_weights()
        
class RNNTokenClassifier(TokenClassifier):
    def __init__(self, config: RNNTokenClassiferConfig, base_model) -> None:
        super().__init__(config, base_model)
        seq_rep_dim = config.sr_dim if config.sr_dim else self.base.config.hidden_size
        self.using_cnn = config.sr_dim is not None
        if self.using_cnn:
            self.cnn = Conv1dModel(config.cnn_layers, config.cnn_layers[-1].out_channels)
            self.seq_rep_mlp = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.BatchNorm1d(config.cnn_layers[-1].out_channels),
                torch.nn.Linear(config.cnn_layers[-1].out_channels, seq_rep_dim),
                torch.nn.ReLU()
            )

            for module in (self.cnn, self.seq_rep_mlp):
                self.init_weights(module)

        self.classifier = RNNClassifier(self.base.config.hidden_size + seq_rep_dim, self.n_labels, config.hidden_size, config.n_layers)
        self.init_weights(self.classifier)

    def append_seq_reps(self, sequence_output, seq_reps):
        seq_reps = seq_reps.unsqueeze(1) # (B, 1, SR_DIM)
        # Repeat the means for every sequence element (i.e. sequence length-times),
        seq_reps= seq_reps.expand(-1, sequence_output.shape[1], -1) # (B, S, SR_DIM)

        return torch.cat([sequence_output, seq_reps], -1) # (B, S, CH + SR_DIM)

    def get_mean_sequence_reps(self, sequence_output : torch.Tensor, batch_lens):
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        pad_mask = torch.arange(0, sequence_output.shape[0], device=self.device)[:, None, None].expand_as(sequence_output)
        lens_reshaped = batch_lens[:, None, None].expand_as(pad_mask)
        # Prepare the mask
        pad_mask = pad_mask > lens_reshaped # True if a given position is padding
        # Zero the padding values
        sequence_output[pad_mask] = 0
        # Zero the BOS token
        sequence_output[:, 0, :] = 0 
        # Calculate the sequence means
        seq_rep = torch.mean(sequence_output, 1)
        # Add the 'sequence' dim
        seq_rep = seq_rep.unsqueeze(1) # (B, 1, CH)
        # Repeat the means for every sequence element (i.e. sequence length-times),
        seq_rep = seq_rep.expand_as(sequence_output) # (B, S, CH)

        return torch.cat([sequence_output, seq_rep], -1) # (B, S, 2CH)
    
    def cnn_features(self, inputs):
        x = self.cnn(inputs)
        x = self.seq_rep_mlp(x)
        return self.append_seq_reps(inputs, x)
    
    def classifier_features(self, inputs, **kwargs):
        if self.using_cnn:
            return self.cnn_features(inputs)
        
        return self.get_mean_sequence_reps(inputs, kwargs['batch_lens'].to(self.device))
    
    def forward(self, input_ids,  attention_mask, batch_lens, **kwargs):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        classifier_features = self.classifier_features(sequence_output, batch_lens=batch_lens)
        return self.classifier(classifier_features, lengths=torch.sum(attention_mask, -1)), outputs
    
class SelectiveFinetuningClassifier(TokenClassifier):
    def __init__(self, config: SelectiveFinetuningClassifierConfig, base_model: Module, ) -> None:
        super().__init__(config, base_model)
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)
        self.init_weights(self.classifier)
        self.set_base_requires_grad(False)
        self.set_indexed_layers_grad(config.unfreeze_indices, True)
        
    def set_indexed_layers_grad(self, indices : list[int], req_grad_value : bool):
        indices = set(indices)
        self.modified_indices = indices
        param_list = list(self.base.encoder.layer.named_children())
        for i in indices:
            # index 0 contains the name, 1 the parameter
            for param in param_list[i][1].parameters():
                param.requires_grad = req_grad_value

class DummyRNNTokenClassifier(TokenClassifier):
    def __init__(self, config: RNNTokenClassiferConfig, base_model) -> None:
        super().__init__(config, base_model)
        print(self.device)

    def forward(self, input_ids, attention_mask, batch_lens, **kwargs):
        return torch.zeros_like(input_ids)
    
    def predict(self, input_ids, attention_mask=None, return_dict=False, labels=None, **kwargs) -> torch.Tensor:
        if labels is not None:
            return torch.Tensor([0], device=self.device), torch.zeros_like(input_ids, device=self.device)
        return torch.zeros_like(input_ids, device=self.device)
    
    def train_predict(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None, return_dict=False, **kwargs):
        self.predict(input_ids, attention_mask, return_dict, labels, kwargs)
