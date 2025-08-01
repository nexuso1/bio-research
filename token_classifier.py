import torch
import torch.nn as nn
import lora
import re
from dataclasses import dataclass
from typing import Callable

@dataclass
class TokenClassifierConfig:
    base_model : nn.Module
    n_labels : int
    lora_config : lora.MultiPurposeLoRAConfig | None = None
    apply_lora : bool = False
    loss : Callable[[torch.Tensor], torch.Tensor]

class TokenClassifier(nn.Module):
    """
    Model that consist of a base embedding model, and a token classification head at the end.
    Can be provided with a sequence representation step override and a LoRA configuration.

    Function set_base_training_status() can freeze/unfreeze the base model, meant to be set 
    by the user before training.
    """
    ignore_index = -100 # Ignore labels with index -100
    token_model = None

    def __init__(self, config : TokenClassifierConfig) -> None:
        super(TokenClassifier, self).__init__()
        self.config = config
        # Embedding model
        self.base = config.base_model

        if config.apply_lora:
            assert config.lora_config is not None
            self.apply_lora(config.lora_config)
            
        
        self.n_labels = config.n_labels
        
        self.loss = config.loss

    def apply_lora(self, config=lora.MultiPurposeLoRAConfig(rank=256)):
        self.lora_config = config
        self.base = lora.modify_with_lora(self.base, self.lora_config)

        # Freeze base model parameters, except LoRA
        for (param_name, param) in self.base.named_parameters():
            param.requires_grad = False       

        for (param_name, param) in self.base.named_parameters():
                if re.fullmatch(self.lora_config.trainable_param_names, param_name):
                    param.requires_grad = True

        print('LoRA applied.')

    def init_weights(self, m):
        """
        Uses xavier/glorot weight initialization for linear layers, and 0 for bias
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def build_classifier(self, args):
        """
        Builds a classifier, connected to the output of the model. By default, 
        it is a simple linear classifier. This method is meant to be overridden
        when implementing other classifiers.

        Do not forget to call self.init_weights after creating the classifier.
        """
        input_size = self.base.config.hidden_size
        self.classifier = torch.nn.Linear(input_size, self.n_labels)

        self.init_weights(self.classifier)

    def classifier_features(self, inputs):
        """
        Function that prepares inputs to the classifier. By default, it does not do anything.
        Meant to be overriden for specific classifier implementations
        """
        return inputs

    def set_base_training_state(self, training_state : bool):
        """
        Freeze/unfreeze the base model according to the training_state bool
        """
        for p in self.base.parameters():
          p.requires_grad = training_state

    def predict(self, input_ids, attention_mask=None, return_dict=False, **kwargs) -> torch.Tensor:
        """
        Prediction in eval mode.
        Outputs are the final classification logits as a Tensor.
        """
        self.eval()
        logits, outputs = self(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if not return_dict:
            return logits
        
        return {
            'logits' : logits,
            'hidden_states' : outputs.hidden_states,
            'attentions' : outputs.attentions,
            'outputs' : outputs
        }

    def train_predict(self, input_ids : torch.Tensor, labels : torch.Tensor, attention_mask : torch.Tensor = None,
                      return_dict=False,  **kwargs):
        """
        Prediction in train mode. Labels should be provided.
        Outputs will contain the loss w.r.t inputs.

        If return_dict is True, output will be a dictionary with additional information about hidden
        states and attentions of the base model
        """
        self.train()
        logits, outputs = self(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.reshape(-1, self.n_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(self.ignore_index).type_as(labels)
            )
            valid_logits=active_logits[active_labels!=-100].flatten()
            valid_labels=active_labels[active_labels!=-100]
            loss = self.loss(valid_logits, valid_labels)
        else:
            loss = self.loss(logits.view(-1, self.n_labels), labels.view(-1))

        if not return_dict:
            return loss, logits
        
        return {
            'loss' : loss,
            'logits' : logits,
            'hidden_states' : outputs.hidden_states,
            'attentions' : outputs.attentions,
            'outputs' : outputs
        }

    def forward(self, input_ids=None, attention_mask=None, **kwargs ):
        """
        Forward pass of the model. Does not make any changes to train/eval status, and does
        not calculate loss.

        Returns (logits, outputs of the base model).
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        classifier_features = self.classifier_features(sequence_output)
        return self.classifier(classifier_features), outputs