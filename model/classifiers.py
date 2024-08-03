from model.token_classifier_base import TokenClassifierConfig
from token_classifier_base import TokenClassifier
from modules import RNNClassifier

import torch

class RNNTokenClassiferConfig(TokenClassifierConfig):
    hidden_size : int
    n_layers : int
    sr_dim : int = None

class RNNTokenClassifer(TokenClassifier):
    def __init__(self, config: RNNTokenClassiferConfig) -> None:
        super().__init__(config)
        seq_rep_dim = config.sr_dim if config.sr_dim else self.base.config.hidden_size
        self.classifier = RNNClassifier(self.base.config.hidden_size + seq_rep_dim, self.n_labels, config.hidden_size, config.n_layers)

    def append_seq_reps(self, sequence_output, seq_reps):
        seq_reps = seq_reps.unsqueeze(1) # (B, 1, SR_DIM)
        # Repeat the means for every sequence element (i.e. sequence length-times),
        seq_reps= seq_reps.expand(-1, sequence_output.shape[1], -1) # (B, S, SR_DIM)

        return torch.cat([sequence_output, seq_reps], -1) # (B, S, CH + SR_DIM)


    def get_mean_sequence_reps(self, sequence_output : torch.Tensor, batch_lens):
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        pad_mask = torch.arange(0, sequence_output.shape[0], device=self.device)[:, None, None].expand_as(sequence_output)
        lens_reshaped = batch_lens[:, None, None].expand_as(pad_mask)
        pad_mask = pad_mask > lens_reshaped # True if a given position is padding
        # Zero the padding values
        sequence_output[pad_mask] = 0
        # Calculate the sequence means
        seq_rep = torch.mean(sequence_output, 1)
        # Add the 'sequence' dim
        seq_rep = seq_rep.unsqueeze(1) # (B, 1, CH)
        # Repeat the means for every sequence element (i.e. sequence length-times),
        seq_rep = seq_rep.expand_as(sequence_output) # (B, S, CH)

        return torch.cat([sequence_output, seq_rep], -1) # (B, S, 2CH)
    
    def classifier_features(self, inputs, attention_mask):
        self.get_mean_sequence_reps(inputs, torch.sum(attention_mask, -1).to(self.device))