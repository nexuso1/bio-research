from esm_train import get_esm
from classifiers import LinearClassifier, TokenClassifierConfig
from training import run_training, parser, create_loss

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    config = TokenClassifierConfig(n_labels=1, loss=create_loss(args))
    model = LinearClassifier(base_model=esm, config=config)
    
    # Freeze the base if we're not using lora (in that case, it is frozen when applying it)
    if not args.lora:
        model.set_base_requires_grad(False)
    return model, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
