# This is the main entrypoint for running UniPTM-based models

from classifiers import UniPTM, TokenClassifierConfig
from esm_train import get_esm
from training import run_training, create_loss, parser

def create_model(args):
    conf = TokenClassifierConfig(1, loss = create_loss(args), base_type=args.type) # ignored
    base, tokenizer = get_esm(conf.base_type)
    classifier = UniPTM(conf, base, 1280, 8, 1, 128, 0.5, 3)
    classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)