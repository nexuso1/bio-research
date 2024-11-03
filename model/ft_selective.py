import ast
from esm import run_training, get_esm, parser, create_loss
from classifiers import SelectiveFinetuningClassifier, SelectiveFinetuningClassifierConfig

def create_model(args):
    esm, tokenizer = get_esm(args.type)
    indices = ast.literal_eval(args.indices)
    config = SelectiveFinetuningClassifierConfig(n_labels=1, loss=create_loss(args), unfreeze_indices=indices)
    model = SelectiveFinetuningClassifier(base_model=esm, config=config)

    return model, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    parser.add_argument('--indices', default="[-1]", default=False, help='Indices of base model layers to be unfrozen')
    args = parser.parse_args()
    main(args)