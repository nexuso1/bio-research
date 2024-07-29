import pandas as pd
import numpy as np
import os
import glob
import baseline
import tensorflow as tf

from argparse import ArgumentParser
from utils import load_tf_model
from utils import load_prot_data
from test_model import analyze_preds, calculate_metrics, flatten_list, save_preds

parser = ArgumentParser()

parser.add_argument('-a', type=bool, help='Analyze mode. Analyze results from an existing result dataframe. The -i argument will then be the dataframe path.', default=False)
parser.add_argument('-i', type=str, help='Model or dataframe path', default='stratified_baseline.h5')
parser.add_argument('-t', type=str, help='Test data path', default='./tfrec_data_residues')
parser.add_argument('--prots', type=str, help='Path to protein dataset, mapping IDs to sequences.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('-p', type=bool, help='Whether the test data are proteins or not', default=True)
parser.add_argument('--max_length', type=int, help='Maximum length of protein sequence to consider (longer sequences will be filtered out of the test data. Default is 1024.', default=1024)

def test_tf_model(args):
    model = load_tf_model(args.i)
    # model = create_eval_model(model)
    paths = glob.glob(f'{args.t}/*.tfrec')

    protein_df = load_prot_data(args.prots).set_index('id') # protein dataset (dataframe)
    #mapping = {id : i for i, id in enumerate(protein_df.index)}
    data = baseline.load_data(paths) # tfrec dataset
    #data = data.map(lambda x: (x['embeddings'], x['target'], x['position'], mapping(x['uniprot_id'].ref()),))
    test = data.batch(4096)
    pred_dict = {id : {'predictions' : np.zeros(shape=(len(protein_df['sequence'][id])))} for id in protein_df.index}
    for batch in test:
        preds = np.argmax(model.predict(batch['embeddings']), axis=-1)
        for i, residue in enumerate(batch['uniprot_id'].numpy().flatten()):
            pred_dict[residue.decode()]['predictions'][int(batch['position'][i])] = preds[i]

    pred_df = pd.DataFrame.from_dict(pred_dict, orient='index')
    test_df = protein_df.join(pred_df)
    print(test_df.head(10))
    analyze_preds(args, test_df)
    calculate_metrics(pred_df['target'], pred_df['target'])

    # Save predictions
    save_preds(args, pred_df)

def create_eval_model(base_model):
    embeds = tf.keras.Input(shape=(1024,), name='embeddings')
    target = tf.keras.Input(shape=(1,), name='target')
    id = tf.keras.Input(shape=(1,), name='uniprot_id')
    position = tf.keras.Input(shape=(1,), name='position')
    preds = base_model(embeds)
    model = tf.keras.Model(inputs=[embeds,  target,  position,  id],
                            outputs={'preds' : preds, 'target' : target, 'position' : position, 'uniprot_id' : id})

    model.summary()
    return model

def main(args):
    test_tf_model(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
