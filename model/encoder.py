from classifiers import EncoderClassifier, EncoderClassifierConfig
from training import run_training
from esm import parser, get_esm, create_loss

def create_model(args):
    mlp_layers = [args.hidden_size for _ in range(args.n_layers_mlp)] + [1]
    conf = EncoderClassifierConfig(1, loss = create_loss(args), mlp_layers=mlp_layers,
                                   n_layers=args.n_layers)
    base, tokenizer = get_esm(conf.base_type)
    classifier = EncoderClassifier(conf, base)

    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    parser.add_argument('--n_layers_mlp', type=int, help='Number of MLP classifier layers', default=3)
    args = parser.parse_args()
    main(args)