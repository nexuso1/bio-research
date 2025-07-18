# This is the main entrypoint for running kinase-aware models

from classifiers import KinaseClassifier, KinaseClassifierConfig
from esm_train import get_esm
from encoder import setup_config
from encoder import add_arguments as add_enc_args
from training import run_training, parser, create_loss


def create_model(args):
    base, tokenizer = get_esm(args.type)
    config = KinaseClassifierConfig(1, loss=create_loss(args), mlp_layers = [], res_cnn_layers=[], sr_cnn_layers=[])
    setup_config(args, config, base.config)
    
    config.kinase_info_path = args.kinase_info_path
    config.kinase_emb_path = args.kinase_emb_path
    config.kinase_transform = args.use_transform
    config.nl_transform = args.nl_transform
    
    classifier = KinaseClassifier(config, base)
    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def add_arguments(parser):
    add_enc_args(parser)
    parser.add_argument('--kinase_info_path', type=str, help='Kinase info csv file path', default='../data/kinases_S.csv')
    parser.add_argument('--kinase_emb_path', type=str,
                        help='Precompute kinase embedding path. Formatted as dict(<id> : <embedding>)',
                        default='../data/kinase_embeddings.pt')
    parser.add_argument('--use_transform', action='store_true', default=False, help='Use a transform layer before prepending kinase embeddings')
    parser.add_argument('--nl_transform', action='store_true', default=False, help='Use a non-linear transform layer before prepending kinase embeddings')
    
def main(args):
    run_training(args, create_model)
    

if __name__ == '__main__':
    add_arguments(parser)

    args = parser.parse_args()
    main(args)