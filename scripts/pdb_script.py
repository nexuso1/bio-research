import urllib
import os
import urllib.parse
import urllib.request
import urllib.error
import json
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-o', type=str, help='Output folder', default='pdbs')
parser.add_argument('-i', type=str, help='Input dataframe with uniprot ids', default='phosphosite_sequences/phosphosite_df.json')
def get_accession_info(id):
    key = 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'
    url = f'https://alphafold.ebi.ac.uk/api/prediction/{id}?key={key}'

    # params = { 'key' : key }
    # data = urllib.parse.urlencode(params)
    #data = data.encode('ascii')
    request = urllib.request.Request(url)
    request.add_header('accept', 'application/json')
    try:
        with urllib.request.urlopen(request) as response:
            res = response.read()
            respone_dict = json.loads(res)

    except urllib.error.HTTPError:
        return False

    return respone_dict

def download_pdb(args, id, url):
    os.makedirs(args.o, exist_ok=True)
    os.system(f'curl {url} -o {args.o}/{id}.pdb')

def process_prots(args):
    prot_df = pd.read_json(args.i).head(2)
    mapping = {}
    for id in prot_df['id']:
        response = get_accession_info(id)
        if not response:
            continue
        pdb_url = response[0]['pdbUrl']
        mapping[id] = response[0]['entryId']
        download_pdb(args, id, pdb_url)
    
    map_df = pd.DataFrame.from_dict(mapping, orient='index', columns=['alphafold_id'])
    map_df.to_json(f'{args.o}/id_mapping_df.json')

def main(args):
    process_prots(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)