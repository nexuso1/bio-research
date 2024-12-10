from classifiers import KinaseClassifierA, KinaseClassifierB
from esm_train import parser
from training import run_training

def create_model(args):
    config = 

def main(args):
    run_training(args, create_model)

if __name__ == '__main__':
    parser.add_argument('--cls_type', default='A', help='Kinase-aware classifier type',type=str)
    parser.add_argument('--n_layers_mlp', type=int, help='Number of MLP classifier layers', default=3)
    parser.add_argument('--block_size', type=int, help='Number of seq. rep. CNN layers in one block', default=2)
    parser.add_argument('--sr_n', type=int, help='Number of seq. rep. CNN blocks', default=3)
    parser.add_argument('--sr_kernel_size', type=int, help='Seq. rep. kernel size', default=5)
    parser.add_argument('--sr_init_size', type=int, help='Initial dimension for the seq. rep. CNN', default=256)
    parser.add_argument('--sr_final_size', type=int, help='Final dimension for the seq. rep. CNN', default=1024)
    parser.add_argument('--encoder_dim', type=int, help='Classifier encoder dimension', default=256)
    args = parser.parse_args()
    main(args)