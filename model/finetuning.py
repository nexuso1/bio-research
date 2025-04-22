from classifiers import SelectiveFinetuningClassifier, SelectiveFinetuningClassifierConfig
from esm_train import get_esm 
from training import create_loss, run_training, parser

def create_model(args):
    loss = create_loss(args)
    indices = [int(i) for i in args.indices]
    config = SelectiveFinetuningClassifierConfig(1, loss=loss, apply_lora=args.lora, dropout_rate=args.droput,
                                                 base_type=args.type, unfreeze_indices=indices)
    base, tokenizer = get_esm(args.type)
    model = SelectiveFinetuningClassifier(config, base)

    return model, tokenizer

def add_arguments(parser):
    parser.add_argument('--indices', type=str, help='Unfreeze given base model layers', default='[-1]')

def main(args):
    run_training(args, create_model_fn=create_model)

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)