import torch
import os
import sys
import datetime

from esm import run_training, get_esm, parser
from classifiers import LinearClassifier, TokenClassifierConfig

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    config = TokenClassifierConfig(n_labels=1, loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])))
    model = LinearClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    # parser.add_argument('--focal_loss', action='store_true', default=False, help='Use focal loss')
    args = parser.parse_args()
    main(args)