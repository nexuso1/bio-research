from torch.nn.modules import Module
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from modules import RNNClassifier
from dataclasses import dataclass, field
from modules import Conv1dModel, ConvLayerConfig, SinPositionalEncoding, ResidualMLP

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
    sr_dim : int = 256
    pos_embed_type : str = 'sin'
    sr_cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 256, 5, 2, 2),
            ConvLayerConfig(256, 378, 5, 2, 2),
            ConvLayerConfig(378, 512, 5, 2, 2),
            ConvLayerConfig(512, 1024, 5, 2, 2),
            # ConvLayerConfig(1024, 768, 3, 2, 2),
            # ConvLayerConfig(768, 512, 3, 2, 2),
            # ConvLayerConfig(512, 384, 3, 2, 2),
            # ConvLayerConfig(384, 256, 3, 3, 2),
        ])
    
    res_cnn_layers : list[ConvLayerConfig] = field(default_factory= lambda :[
            ConvLayerConfig(1280, 256, 31, 4, 1),
        ])
    mlp_layers : list[int] = field(default_factory=lambda : [256, 256, 256, 1])

@dataclass
class SelectiveFinetuningClassifierConfig(TokenClassifierConfig):
    unfreeze_indices : list[int] = field(default_factory= lambda : [-1])

class LinearClassifier(TokenClassifier):
    def __init__(self, config: TokenClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)
        self.init_weights(self.classifier)

class EncoderClassifier(TokenClassifier):
    def __init__(self, config: EncoderClassifierConfig, base_model: Module) -> None:
        super().__init__(config, base_model)
        enc_layer = torch.nn.TransformerEncoderLayer(config.sr_dim, nhead=config.n_heads,
                                                    dim_feedforward=config.hidden_size,
                                                    activation='relu', batch_first=True)
        
        # Setup positional embeddings
        if config.pos_embed_type == 'sin':
            self.pos_embed = SinPositionalEncoding(config.sr_dim, 1026) # [seq_rep][cls]...[eos]
        # elif config.pos_embed_type == 'rope':
        #     self.pos_embed = RotaryPositionalEmbeddings(config.sr_dim // config.n_heads, 1024)
        else:
            self.pos_embed = None

        self.encoder = torch.nn.TransformerEncoder(enc_layer, config.n_layers)

        # Create sequence-representation and residue-representation CNNs
        self.sr_cnn = Conv1dModel(config.sr_cnn_layers, config.sr_cnn_layers[-1].out_channels, self.config.dropout_rate)
        self.res_cnn = Conv1dModel(config.res_cnn_layers, config.sr_cnn_layers[-1].out_channels, False, self.config.dropout_rate)

        # Create the FFN parts of these representations
        self.seq_rep = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LayerNorm(config.sr_cnn_layers[-1].out_channels),
                torch.nn.Linear(config.sr_cnn_layers[-1].out_channels, config.sr_dim),
                torch.nn.ReLU()
            )

        # Create a residual MLP classifier
        self.classifier = ResidualMLP(self.config.mlp_layers,
                                       input_size=config.sr_dim, activation=torch.nn.ReLU(), norm=torch.nn.LayerNorm,
                                       dropout=config.dropout_rate)

        # Initialize the modules
        for module in self.modules():
            module.apply(self.xavier_init)
        
        self.base = base_model
    
    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        x = base_out[0]
        proj = self.res_cnn(x)
        seq_rep = self.sr_cnn(x)
        seq_rep = self.seq_rep(seq_rep)
        enc_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, device=self.device), attention_mask], 1)
        x = torch.cat([seq_rep.unsqueeze(1), proj], axis=1)
        x = x + self.pos_embed(x)
        x = self.encoder(x, src_key_padding_mask=torch.bitwise_not(enc_mask.bool()))
        return self.classifier(x)[:, 1:], base_out

class UniPTM(TokenClassifier):
    def __init__(self, config, base, emb_size, num_heads, num_layers, hidden_size, dropout_rate, pos_weight=None):
        super(UniPTM, self).__init__(base_model=base, config=config)
        self.cnn = torch.nn.Conv1d(in_channels=emb_size, out_channels=256, kernel_size=31, padding=15) 
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.fc1 = torch.nn.Linear(256, hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
        self.pos_weight = pos_weight

        self.base = base
    
    def forward(self, input_ids, attention_mask, **kwargs):
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        emb = base_out[0]
        emb = emb.transpose(1, 2)  
        emb = self.cnn(emb)
        emb = emb.transpose(1, 2)  
        x = self.encoder(emb)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x, base_out
    
    def train_predict(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None, return_dict=False, **kwargs):
        self.train()
        out = self(input_ids, attention_mask)[0]
        return self.weighted_BCEloss(labels != -1, labels, torch.sigmoid(out)), out 

    def weighted_BCEloss(self, mask, labels, outputs):
        mask = mask.squeeze(0).bool()
        true_y = labels.squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        weights = torch.ones_like(true_y)  
        if self.pos_weight is not None:
            weights[true_y == 1] = self.pos_weight  
        loss = torch.nn.functional.binary_cross_entropy(pred_y, true_y, weight=weights)
        return loss

    def BCEloss(self, batch, outputs):
        mask = batch['mask']
        mask = mask.squeeze(0).bool()
        true_y = batch['label'].squeeze(0)[mask].float()
        pred_y = outputs.squeeze(0)[mask].squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(pred_y, true_y)
        return loss

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

class DummyClassifier(TokenClassifier):
    def __init__(self, config: RNNTokenClassiferConfig, base_model) -> None:
        super().__init__(config, base_model)
        self.linear = torch.nn.Linear(1, 1) # So that the parameter list for optim isn't empty

    def forward(self, input_ids, attention_mask, batch_lens, **kwargs):
        return torch.zeros_like(input_ids, device=self.device).float()
    
    def predict(self, input_ids, attention_mask=None, return_dict=False, labels=None, **kwargs) -> torch.Tensor:
        preds = self.linear(input_ids.float().unsqueeze(-1))
        preds = preds - preds
        if labels is not None:
            return self.loss(preds.squeeze(), labels), preds
        return preds
    
    def train_predict(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None, return_dict=False, **kwargs):
        return self.predict(input_ids, attention_mask, return_dict, labels, **kwargs)
