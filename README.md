# Prediction of Phosphorylation Sites using Protein Language Models

This is the main repository containing source code and data used in our project.

``data`` contains both our dataset splits as well as the data obtained from the UniPTM repository and Kinase Library repository.
- Residue datasets located in ``data/splits_*.json``.
- These contain indices into ``data/phosphosite_sequences/phosphosite_df.json``. The files themselves are split into 5 lists, one for each fold.

- Dataset are created at runtime using the indices from the split file

``model`` contains the source code for our models and the training loop

``notebooks`` contains various JuPyter notebooks for the creation of visualizations, data gathering a dataset preparation.


