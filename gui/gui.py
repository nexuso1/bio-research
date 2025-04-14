import streamlit as st
import os
import torch
import argparse
import pandas as pd
import sys
import plotly.express as pt
import numpy as np

sys.path.append('./model') 

device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

def load_model(filepath):
    """Loads a model from the given filepath."""
    try:
        checkpoint = torch.load(filepath, map_location=device)

    except Exception as e:
        st.error(f"Error loading model from {filepath}: {e}")
        return None, None
    
    args = argparse.Namespace()
    for k, v in checkpoint['hyper_parameters'].items():
        args.__setattr__(k, v)

    try:
        model, tokenizer = create_kinase_model(args)
        print('kinase model loaded')
    except AttributeError:
        model, tokenizer = create_encoder_model(args)
        print('encoder model loaded')

    # Fix potential dictionary issues with checkpoint keys
    prefix = 'classifier.'
    for k in list(checkpoint['state_dict'].keys()):
        checkpoint['state_dict'][k.removeprefix(prefix)] = checkpoint['state_dict'].pop(k)
    model.load_state_dict(checkpoint['state_dict'])
    return model, tokenizer


import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def get_text_color_for_bg(bg_color_hex):
    """
    Determines if text should be black or white based on background color luminance.
    Args:
        bg_color_hex (str): Background color in hex format (e.g., '#RRGGBB').
    Returns:
        str: 'black' or 'white'.
    """
    # Convert hex to RGB
    rgb = mcolors.hex2color(bg_color_hex)
    # Calculate luminance (using standard formula)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    # Return black for light backgrounds, white for dark backgrounds
    return 'black' if luminance > 0.5 else 'white'

def display_colored_residues(sequence, probabilities, pred_residue, cmap_name='viridis', prob_label="Probability"):
    """
    Displays a sequence with residues colored by probability in Streamlit.

    Args:
        sequence (str): The sequence of residues (e.g., protein sequence).
        probabilities (list or np.array): A list or array of probabilities,
                                          one for each residue, in the range [0, 1].
        cmap_name (str): Name of the Matplotlib colormap to use (e.g., 'viridis', 'coolwarm').
        prob_label (str): Label to use for the color bar.
    """
    if len(sequence) != len(probabilities):
        st.error(f"Error: Sequence length ({len(sequence)}) and probabilities length ({len(probabilities)}) must match.")
        return

    # --- 1. Generate HTML for colored sequence ---
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1) # Normalize probabilities to 0-1 range for colormap

    html_elements = []
    for i, residue, prob in zip(range(len(sequence) ), sequence, probabilities):
        # Clamp probability just in case it's slightly outside [0, 1]
        if residue == pred_residue:
            rgba_color = cmap(norm(prob))
            #text_color = get_text_color_for_bg(hex_color)
            # Add a tooltip showing the exact probability on hover
            tooltip = f"Prob: {prob:.3f}\nIndex: {i}"
        else:
            rgba_color = [0.4, 0.4, 0.4] # grey
            tooltip = f"Non-matching residue"

        hex_color = mcolors.to_hex(rgba_color)
        # Style each residue individually
        # Added slight padding and margin for better visual separation
        html_elements.append(
            f'<span title="{tooltip}" style="'
            #f'background-color: {hex_color}; '
            f'color: {hex_color}; '
            f'font-size: 20px; '
            f'padding: 1px 2px; ' # Minimal padding
            f'margin: 0 1px; ' # Minimal margin
            f'border-radius: 3px;' # Slightly rounded corners
            f'">{residue}</span>'
        )

    # Join all spans into a single HTML string
    full_html = f'<div style="line-height: 1.5; font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_elements)}</div>'

    # Display the HTML in Streamlit
    st.markdown("### Colored Sequence:")
    st.markdown(full_html, unsafe_allow_html=True)

    

def predict_sequence(model, tokenizer, sequence) -> np.ndarray:
    """Makes predictions on the input sequence using the loaded model.

    Args:
        model: The loaded token classification model.
        tokenizer: The tokenizer for this model.
        sequence (str): The input sequence.

    Returns:
        probs: A numpy array of prediction probabilites for each token.
    """
    if model:
        model_input = tokenizer(sequence, return_tensors='pt')
        logits = model.predict(**model_input.to(device)).squeeze()
        probs = torch.sigmoid(logits)
        probs = probs[1:-1] # Remove [BOS], [EOS]
        return probs.cpu().numpy()
    return None

def format_predictions(predictions, residue : str, sequence : str) -> pd.DataFrame:
    """Formats predictions for row-wise display."""
    if predictions is None:
        return None

    mask = [t.upper() == residue for t in sequence]
    probabilities = [p for p in predictions[mask]]
    indices = np.arange(len(sequence))[mask].tolist()

    formatted_data = {
        "Probability": probabilities,
        "Sequence Index": indices
    }
    return pd.DataFrame.from_dict(formatted_data, orient='columns').set_index('Sequence Index')

# --- Streamlit App ---

st.title("Token Classification Model Interface")

from encoder import create_model as create_encoder_model
from kinase import create_model as create_kinase_model

# Directory containing your model files
model_directory = "./gui/models"

# Get a list of model files in the directory
model_files = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f))]

if not model_files:
    st.warning(f"No model files found in the '{model_directory}' directory.")
else:
    # Dropdown to select the model
    selected_model_file = st.selectbox("Select a Model:", model_files)
    residue = st.selectbox('Select a residue to predict (should match what the model was trained on):', ['S', 'T', 'Y'])
    model_path = os.path.join(model_directory, selected_model_file)
    loaded_model, tokenizer = load_model(model_path)

    if loaded_model:
        # Input text area for the sequence
        input_sequence = st.text_area("Enter Sequence:", "")
        input_sequence = input_sequence.upper().strip()

        # Predict button
        if st.button("Predict"):
            if input_sequence:
                predictions = predict_sequence(loaded_model, tokenizer, input_sequence)
                if predictions is not None:
                    formatted_output = format_predictions(predictions, residue, input_sequence)
                    display_colored_residues(input_sequence, predictions, residue,
                                              cmap_name='plasma', prob_label="Predicted Probability",)
                    st.subheader("Predictions table:")
                    chart = st.table(formatted_output)
                else:
                    st.warning("No predictions generated.")
            else:
                st.warning("Please enter a sequence to predict.")

# --- Instructions ---
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1.  Place your trained token classification model files (e.g., `.pkl` files) in a folder named 'models'.
    2.  Select a model from the dropdown menu.
    3.  Enter the sequence you want to classify in the text area.
    4.  Click the 'Predict' button.
    5.  The sequence residues will be colored according to the predicted probabilities. Brighter colors correspond to higher probabilities.
    6.  The predictions will also be displayed in a table format with sequence indices and corresponding probabilities as rows.

    You can hover over the colored text to see the probabilites and sequence indices.
    """
)