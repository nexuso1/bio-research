from classifiers import EncoderClassifier, EncoderClassifierConfig, ConvLayerConfig, FusedMBConvConfig
from argparse import ArgumentParser
from training import run_training, parser, create_loss
from esm_train import get_esm
import numpy as np

def create_model(args):
    base, tokenizer = get_esm(args.type)
    mlp_layers = [args.hidden_size for _ in range(args.n_layers_mlp)] + [1]
    sr_sizes = 2 ** np.linspace(np.log2(args.sr_init_size), np.log2(args.sr_final_size), args.sr_n)
    base_size = base.config.hidden_size

    if args.cnn_type == 'basic':
        # First layer
        sr_cnn_layers = [ConvLayerConfig(base_size, int(sr_sizes[0]), args.sr_kernel_size, args.block_size, 2)]
        # Rest of layers
        sr_cnn_layers = sr_cnn_layers + \
            [ConvLayerConfig(int(sr_sizes[i]), int(sr_sizes[i+1]), args.sr_kernel_size, args.block_size, 2)
            for i in range(args.sr_n - 1)]
        
        # Should only have one layer
        res_cnn_layers = [ConvLayerConfig(base_size, args.encoder_dim, args.res_kernel_size, 1, 1)]

    elif args.cnn_type == 'fused':
        sr_cnn_layers = [FusedMBConvConfig(base_size, int(sr_sizes[0]), args.sr_kernel_size, args.block_size, 2, args.expand)]

        sr_cnn_layers = sr_cnn_layers + \
            [FusedMBConvConfig(int(sr_sizes[i]), int(sr_sizes[i+1]), args.sr_kernel_size, args.block_size, 2, args.expand) 
            for i in range(args.sr_n - 1)]
        res_cnn_layers = [FusedMBConvConfig(base_size, args.encoder_dim, args.res_kernel_size, 1, 1, args.expand)]
    
    conf = EncoderClassifierConfig(1, loss = create_loss(args), mlp_layers=mlp_layers,
                                    n_layers=args.n_layers, dropout_rate=args.dropout,
                                    sr_dim=args.encoder_dim,
                                    sr_cnn_layers=sr_cnn_layers,
                                    sr_type=args.sr_type,
                                    res_cnn_layers=res_cnn_layers,
                                    cnn_type=args.cnn_type,
                                    encoder_dim=args.encoder_dim)
    
    classifier = EncoderClassifier(conf, base)
 
    if not args.lora:
        classifier.set_base_requires_grad(False)

    return classifier, tokenizer

def add_arguments(parser : ArgumentParser):
    parser.add_argument('--n_layers_mlp', type=int, help='Number of MLP classifier layers', default=3)
    parser.add_argument('--block_size', type=int, help='Number of seq. rep. CNN layers in one block', default=2)
    parser.add_argument('--cnn_type', type=str, help=
                        '''Type of cnn to be used for seq/residue reps. Options are "basic",
                        for standard Conv1d layers, or "fused", for FusedMBConv layers''',
                        default='basic')
    parser.add_argument('--expand_m', type=int, help='If using FusedMBConv, this is the expansion multiplier.', default=4)
    parser.add_argument('--sr_n', type=int, help='Number of seq. rep. CNN blocks', default=3)
    parser.add_argument('--sr_kernel_size', type=int, help='Seq. rep. kernel size', default=5)
    parser.add_argument('--sr_init_size', type=int, help='Initial dimension for the seq. rep. CNN', default=256)
    parser.add_argument('--sr_final_size', type=int, help='Final dimension for the seq. rep. CNN', default=512)
    parser.add_argument('--res_kernel_size', type=int, help='Residue representation kernel size', default=31)
    parser.add_argument('--sr_type', type=str, help='Sequence representation type. Either "mean" or "cnn".', default='cnn')
    parser.add_argument('--encoder_dim', type=int, help='Classifier encoder dimension', default=256)
    parser.add_argument('--expand', type=int, help='Expansion constant for fused mbconv', default=4)

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    add_arguments(parser)
    args = parser.parse_args()
    main(args)