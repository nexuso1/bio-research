from classifiers import EncoderClassifier, EncoderClassifierConfig, ConvLayerConfig
from training import run_training
from esm import parser, get_esm, create_loss
import numpy as np

def create_model(args):
    base, tokenizer = get_esm(args.type)
    mlp_layers = [args.hidden_size for _ in range(args.n_layers_mlp)] + [1]
    sr_sizes = 2 ** np.linspace(np.log2(args.sr_init_size), np.log2(args.sr_final_size), args.sr_n)
    sr_cnn_layers = [ConvLayerConfig(base.config.hidden_size, sr_sizes[0], args.sr_kernel_size, args.block_size, 2)]
    sr_cnn_layers = sr_cnn_layers + [ConvLayerConfig(sr_sizes[i], sr_sizes[i+1], args.sr_kernel_size, args.block_size, 2) for i in range(args.sr_n - 1)]
    print(sr_cnn_layers)
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
    parser.add_argument('--block_size', type=int, help='Number of seq. rep. CNN layers in one block', default=2)
    parser.add_argument('--sr_n', type=int, help='Number of seq. rep. CNN blocks', default=3)
    parser.add_argument('--sr_kernel_size', type=int, help='Seq. rep. kernel size', default=5)
    parser.add_argument('--sr_init_size', type=int, help='Initial dimension for the seq. rep. CNN', default=256)
    parser.add_argument('--sr_final_size', type=int, help='Final dimension for the seq. rep. CNN', default=1024)
    parser.add_argument('--encoder_dim', type=int, help='Classifier encoder dimension', default=256)
    args = parser.parse_args()
    main(args)