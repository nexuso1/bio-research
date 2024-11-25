from classifiers import EncoderClassifier, EncoderClassifierConfig
from training import run_training
from esm import parser, get_esm, create_loss

def create_model(args):
    conf = EncoderClassifierConfig(1, loss = create_loss(args))
    base, tokenizer = get_esm(conf.base_type)
    classifier = EncoderClassifier(conf, base)

    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)