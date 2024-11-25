from classifiers import UniPTM, TokenClassifierConfig
from esm import create_loss, get_esm, parser
from training import run_training

def create_model(args):
    conf = TokenClassifierConfig(1, loss = create_loss(args)) # ignored
    base, tokenizer = get_esm(conf.base_type)
    classifier = UniPTM(base, 1280, 8, 1, 128, 0.5, 3)

    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)