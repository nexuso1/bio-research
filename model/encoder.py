from classifiers import EncoderClassifier, EncoderClassifierConfig, ConvLayerConfig
from training import run_training
from esm import parser, get_esm, create_loss

def create_model(args):
    base, tokenizer = get_esm(args.type)
    mlp_layers = [args.hidden_size for _ in range(args.n_layers_mlp)] + [1]
    sr_cnn_layers = [ConvLayerConfig(base.config.hidden_size, args.encoder_dim, args.sr_kernel_size, args.block_size, 2)]
    sr_cnn_layers = sr_cnn_layers + [ConvLayerConfig(args.encoder_dim, args.encoder_dim, args.sr_kernel_size, args.block_size, 2) for _ in range(args.n_blocks - 1)]
    conf = EncoderClassifierConfig(1, loss = create_loss(args), mlp_layers=mlp_layers,
                                   n_layers=args.n_layers, dropout_rate=args.dropout,
                                     sr_cnn_layers=sr_cnn_layers)
    
    classifier = EncoderClassifier(conf, base)
 
    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    parser.add_argument('--n_layers_mlp', type=int, help='Number of MLP classifier layers', default=3)
    parser.add_argument('--block_size', type=int, help='Number of seq. rep. CNN layers in one block', default=3)
    parser.add_argument('--n_blocks', type=int, help='Number of seq. rep. CNN blocks', default=3)
    parser.add_argument('--sr_kernel_size', type=int, help='Seq. rep. kernel size')
    parser.add_argument('--encoder_dim', type=int, help='Classifier encoder dimension', default=256)
    args = parser.parse_args()
    main(args)