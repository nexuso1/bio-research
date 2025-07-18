# Entrypoint for the Dummy models, which were sometimes used to 
# debug the training process

from classifiers import DummyClassifier, TokenClassifierConfig
from transformers import AutoTokenizer
from training import parser, create_loss, run_training

def create_model(args):
    config = TokenClassifierConfig(1, create_loss(args))
    model = DummyClassifier(config, None)
    return model, AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')

if __name__ == "__main__":
    args = parser.parse_args()
    run_training(args, create_model)