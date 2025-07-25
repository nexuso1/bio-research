"""
https://github.com/jasperzuallaert/PhosphoLingo/blob/master/phospholingo/input_reader.py
"""

import numpy as np

class TokenAlphabet(object):
    """
    Parent class for token representations of different protein language models. Code is based on the ESM implementation
    at https://github.com/facebookresearch/esm/.

    Attributes
    ----------
    pre_idx_tokens : list[str]
        The tokens that have indices starting from 0, before the actual amino acid tokens

    amino_acid_tokens : list[str]
        The amino acid tokens in a predetermined order

    post_idx_tokens : list[str]
        The tokens that have indices succeeding the amino acid tokens

    all_tokens : list[str]
        Combination of pre_idx_tokens, amino_acid_tokens, post_idx_tokens

    seq_prepend_tokens : list[str]
        Extra tokens that should be added at the front of a protein sequence

    seq_append_tokens : list[str]
        Extra tokens that should be added at the back of a protein sequence

    tok_to_idx : dict[str,int]
        Token indices for all amino acid and extra tokens
    """

    # set all tokens: amino acids, and any tokens with indices before or after the amino acids
    pre_idx_tokens = ["<pad>", "<unk>"]
    amino_acid_tokens = list("ACDEFGHIKLMNPQRSTVWY")
    post_idx_tokens = []
    all_tokens = []

    # set which tokens should be added in front/at the back of each encoded sequence
    seq_prepend_tokens = []
    seq_append_tokens = []

    tok_to_idx = {}

    def __len__(self) -> int:
        return len(self.tok_to_idx)

    def tokenize(self, seq: str) -> np.ndarray:
        """
        Converts a string representation of a sequence to its token representation

        Parameters
        ----------
        seq : str
            The input sequence to be converted

        Returns
        -------
        token_ids : np.ndarray
            Token representation of the input sequence, including extra tokens at front and back if specified
        """
        token_ids = (
            [self.tok_to_idx[x] for x in self.seq_prepend_tokens]
            + [self.tok_to_idx[x if x in self.tok_to_idx else "<unk>"] for x in seq]
            + [self.tok_to_idx[x] for x in self.seq_append_tokens]
        )
        return np.array(token_ids, np.int64)

    # returns the mask for the actual amino acid tokens only
    def get_mask_no_extra(self, seq: str) -> np.ndarray:
        """
        Returns a mask for a tokenized sequence, only including the amino acids (i.e. extra tokens get zeros)

        Parameters
        ----------
        seq : str
            The string representation of the input sequence

        Returns
        -------
        : np.ndarray
            A mask with 1's at amino acid positions, and 0's at extra tokens
        """
        return np.array(
            [0 for _ in self.seq_prepend_tokens]
            + [1 for _ in seq]
            + [0 for _ in self.seq_append_tokens]
        )

    def get_num_tokens_added(self) -> int:
        return len(self.seq_prepend_tokens) + len(self.seq_append_tokens)

    def get_num_tokens_added_front(self) -> int:
        return len(self.seq_prepend_tokens)

    def get_num_tokens_added_back(self) -> int:
        return len(self.seq_append_tokens)


class OneHotAlphabet(TokenAlphabet):
    pre_idx_tokens = []
    amino_acid_tokens = list("ACDEFGHIKLMNPQRSTVWY")
    post_idx_tokens = ["<unk>", "<pad>"]
    seq_prepend_tokens = []
    seq_append_tokens = []

    def __init__(self):
        """
        Tokenizer for a classical one-hot encoding
        """
        self.all_tokens = list(self.pre_idx_tokens)
        self.all_tokens.extend(self.amino_acid_tokens)
        self.all_tokens.extend(self.post_idx_tokens)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_tokens)}


# ESM1-small and ESM1b model encoding
class ESMAlphabet(TokenAlphabet):
    prepend_tokens = ("<null_0>", "<pad>", "<eos>", "<unk>")
    standard_tokens = list("LAGVSERTIDPKQNFYMHWCXBUZO")
    append_tokens = ("<cls>", "<mask>", "<sep>")
    seq_prepend_tokens = ["<cls>"]
    seq_append_tokens = []

    def __init__(self):
        """
        Tokenizer for the ESM language models (https://github.com/facebookresearch/esm/)
        """
        # this block is from the ESM code
        self.all_tokens = list(self.prepend_tokens)
        self.all_tokens.extend(self.standard_tokens)
        for i in range((8 - (len(self.all_tokens) % 8)) % 8):
            self.all_tokens.append(f"<null_{i  + 1}>")
        self.all_tokens.extend(self.append_tokens)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_tokens)}


class ProtTransAnkhAlphabet(TokenAlphabet):
    prepend_tokens = ("<pad>", "<sep>", "<unk>")
    standard_tokens = list("ALGVSREDTIPKFQNYMHWCXBOUZ")
    append_tokens = ()
    seq_prepend_tokens = []
    seq_append_tokens = []

    def __init__(self):
        """
        Tokenizer for the ProtTrans and Ankh language models (https://github.com/agemagician/ProtTrans, https://github.com/agemagician/Ankh)
        """
        self.all_tokens = list(self.prepend_tokens)
        self.all_tokens.extend(self.standard_tokens)
        self.all_tokens.extend(self.append_tokens)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_tokens)}

class CARPAlphabet(TokenAlphabet):
    prepend_tokens = ()
    standard_tokens = list('ACDEFGHIKLMNPQRSTVWYBZXJOU*')#-#@
    append_tokens = ('<pad>', '<unk>', '<mask>')
    seq_prepend_tokens = []
    seq_append_tokens = []

    def __init__(self):
        """
        Tokenizer for the CARP language model (https://github.com/microsoft/protein-sequence-models)
        """
        self.all_tokens = list(self.prepend_tokens)
        self.all_tokens.extend(self.standard_tokens)
        self.all_tokens.extend(self.append_tokens)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_tokens)}
        self.set_indices()
        self.tok_to_idx['?'] = self.unk_idx

    def set_indices(self):
        self.unk_idx = self.tok_to_idx["<unk>"] if '<unk>' in self.tok_to_idx else None
        self.padding_idx = self.tok_to_idx["<pad>"] if '<pad>' in self.tok_to_idx else None
        self.cls_idx = self.tok_to_idx["<cls>"] if '<cls>' in self.tok_to_idx else None
        self.mask_idx = self.tok_to_idx["<mask>"] if '<mask>' in self.tok_to_idx else None
        self.eos_idx = self.tok_to_idx["<eos>"] if '<eos>' in self.tok_to_idx else None
        self.sep_idx = self.tok_to_idx["<sep>"] if '<sep>' in self.tok_to_idx else None


tokenizers = {
  'onehot': OneHotAlphabet,
  'ESM1_small': ESMAlphabet,
  'ESM1b': ESMAlphabet,
  'ESM-1b': ESMAlphabet,
  'ESM2_150M': ESMAlphabet,
  'ESM2_650M': ESMAlphabet,
  'ESM2_3B': ESMAlphabet,
  'ESM2_15B': ESMAlphabet,
  'CARP_640M': CARPAlphabet,
  'ProtTransT5_XL_UniRef50': ProtTransAnkhAlphabet,
  'Ankh_base': ProtTransAnkhAlphabet,
  'Ankh_large': ProtTransAnkhAlphabet
}
