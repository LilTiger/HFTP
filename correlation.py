import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel,  AutoTokenizer, AutoModelForCausalLM
import h5py
from scipy.fft import fft
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, resample
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
import ast
import logging
import itertools
from scipy.stats import chisquare
epsilon = 1e-10

def calculate_itpc(phases):
    """Calculate Inter-Trial Phase Coherence (ITPC)."""
    return np.abs(np.sum(np.exp(1j * phases), axis=0)) / phases.shape[0]


def process_seeg(sub_folder, sub_id, region_df, output_dir='Results/correlation/permutations'):
    """Process sEEG data for a subject and calculate ITPC for each channel."""
    import re

    # Check and remove existing HDF5 files if needed
    for hemisphere in ['L', 'R']:
        output_file = f"{output_dir}/itpc_{hemisphere}.hdf5"
        if sub_id == 1 and os.path.exists(output_file):
            print(f"Removing existing file: {output_file}")
            os.remove(output_file)

    results = {'L': {}, 'R': {}}

    def format_label(label, sub_id):
        """
        Format a label string according to the subject-specific rules:
        - For general cases:
          'B 01-Ref' → 'B01'
          'P'' 09-Ref' → 'P'09'
        - For sub_id == 6: Replace multiple spaces with "'"
        - For sub_id == 16: Handle 'A1' and 'B''1' separately.
        """
        if isinstance(label, np.ndarray):
            label = label[0]

        if isinstance(label, bytes):
            label = label.decode('utf-8')

        if sub_id == 6:
            label = re.sub(r"\s+", "'", label)
            match = re.match(r"([A-Z]')(\d+)-", label)
            if match:
                return f"{match.group(1)}{match.group(2)}"
        elif sub_id == 16:
            if re.match(r"^[A-Z]\d+$", label):  # Matches 'A1'
                return label
            if re.match(r"^[A-Z]'\d+$", label):  # Matches 'B'1'
                return label

        # General case
        match = re.match(r"([A-Z])\s?(\d+)-", label)
        if match:
            return f"{match.group(1)}{match.group(2)}"

        match_with_apostrophe = re.match(r"([A-Z]''?)\s?(\d+)-", label)
        if match_with_apostrophe:
            formatted_label = match_with_apostrophe.group(1).replace("''", "'") + match_with_apostrophe.group(2)
            return formatted_label

        raise ValueError(f"Unexpected label format: {label}")

    # Define condition to block mapping for different experimental conditions
    condition_block_map = {
        'sentence': ['Sentence_block1', 'Sentence_block2'],
        'phrase': ['Phrase_block1', 'Phrase_block2'],
        'rand': ['Rand_block1', 'Rand_block2']
    }

    for condition in ['rand', 'sentence', 'phrase']:
        if sub_id == 21 and condition == 'phrase':
            continue

        file_path = os.path.join(sub_folder, f'EEG_{condition}_chanlocs_atlas_data.mat')
        if not os.path.exists(file_path):
            continue

        mat_data = scipy.io.loadmat(file_path)
        struct_name = [key for key in mat_data.keys() if not key.startswith('__')][0]
        data_struct = mat_data[struct_name]

        # Decode labels and data
        labels = [label[0] if isinstance(label, np.ndarray) else label for label in data_struct['labels']]
        data = [d[0] for d in data_struct['data']]
        aal_name = [x[0][0] for x in data_struct['aal_name']]

        aal_name_df = pd.DataFrame(aal_name, columns=['aal_name'])
        merged_df = aal_name_df.merge(region_df, how='left', on='aal_name')
        valid_indices = merged_df['region_index'].notnull()
        valid_data_indices = np.where(valid_indices)[0]

        # Filter data
        filtered_data = [data[i] for i in valid_data_indices]
        region_label = merged_df.loc[valid_indices, 'region_label'].values
        region_index = merged_df.loc[valid_indices, 'region_index'].values

        # Determine left and right indices (within the filtered set)
        left_indices = [i for i, lbl in enumerate(region_label) if lbl.endswith('_L')]
        right_indices = [i for i, lbl in enumerate(region_label) if lbl.endswith('_R')]

        # Concatenate data for left and right channels
        left_data = np.concatenate([filtered_data[i] for i in left_indices], axis=0) if left_indices else np.array([])
        right_data = np.concatenate([filtered_data[i] for i in right_indices], axis=0) if right_indices else np.array(
            [])

        srate = data_struct['srate'][0, 0][0, 0]
        if sub_id == 16:
            left_data = left_data[:, 2048:, :] if left_data.size > 0 else left_data
            right_data = right_data[:, 2048:, :] if right_data.size > 0 else right_data
        else:
            left_data = left_data[:, 512:, :] if left_data.size > 0 else left_data
            right_data = right_data[:, 512:, :] if right_data.size > 0 else right_data

        # Process each hemisphere
        for hemisphere, hemi_data, hemi_indices in zip(['L', 'R'], [left_data, right_data],
                                                       [left_indices, right_indices]):
            if hemi_data.size == 0:
                continue

            n_trials = hemi_data.shape[2]
            trial_split = [(0, n_trials // 2), (n_trials // 2, n_trials)] if n_trials >= 20 else [(0, n_trials)]

            for block_idx, (start, end) in enumerate(trial_split, 1):
                block_data = hemi_data[:, :, start:end]
                itpc_results = np.zeros((block_data.shape[0], block_data.shape[1] // 2 + 1))

                # Compute ITPC
                for ch_idx in range(block_data.shape[0]):
                    phases = []
                    for trial in range(block_data.shape[2]):
                        trial_fft = fft(block_data[ch_idx, :, trial])
                        phases.append(np.angle(trial_fft[:block_data.shape[1] // 2 + 1]))
                    itpc = calculate_itpc(np.array(phases))
                    itpc_results[ch_idx, :] = itpc

                # Save each channel's results
                for idx_in_filtered, filtered_data_ch_idx in enumerate(hemi_indices):
                    # Map back to original index
                    original_idx = valid_data_indices[filtered_data_ch_idx]

                    region = region_label[filtered_data_ch_idx]
                    region_idx = region_index[filtered_data_ch_idx]

                    # Format label based on original index
                    xxx = format_label(labels[original_idx], sub_id)
                    channel_key = f"sub{sub_id}_{xxx}"

                    # Check hemisphere naming convention
                    # According to the user's requirement:
                    # For the left hemisphere (hemisphere == 'L'), the channel label should NOT contain "'"
                    # For the right hemisphere (hemisphere == 'R'), the channel label SHOULD contain "'"
                    if hemisphere == 'L' and "'" in xxx:
                        raise ValueError(
                            f"Hemisphere naming error: {channel_key} contains apostrophe but is in left hemisphere")
                    if hemisphere == 'R' and "'" not in xxx:
                        raise ValueError(
                            f"Hemisphere naming error: {channel_key} does not contain apostrophe but is in right hemisphere")

                    if (hemisphere == 'L' and region.endswith('_L')) or (hemisphere == 'R' and region.endswith('_R')):
                        output_file = f"{output_dir}/itpc_{hemisphere}.hdf5"
                        block_name = condition_block_map[condition][block_idx - 1]

                        with h5py.File(output_file, 'a') as f:
                            if channel_key not in f:
                                f.create_group(channel_key)
                            if block_name in f[channel_key]:
                                del f[channel_key][block_name]
                            # idx_in_filtered maps to itpc_results row
                            f[channel_key].create_dataset(block_name, data=itpc_results[idx_in_filtered, :])
                            f[channel_key].attrs['region_label'] = region
                            f[channel_key].attrs['region_index'] = region_idx
                    else:
                        print(f"Error: Region {region} does not match hemisphere {hemisphere}")

    return results


def gpt2_activation(corpus_path, model_name='gpt2-large-chinese', output_dir='Results/correlation/activations'):
    """Extract neural activations from GPT-2 model for the given corpus."""
    tokenizer = BertTokenizer.from_pretrained(f"./models/gpt/chinese/{model_name}")
    config = GPT2Config.from_pretrained(f"./models/gpt/chinese/{model_name}", output_hidden_states=True)
    model = GPT2LMHeadModel.from_pretrained(f"./models/gpt/chinese/{model_name}", torch_dtype=torch.float16, config=config).to('cuda')

    # Load the corpus
    corpus_df = pd.read_csv(corpus_path, header=None)
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # Define block divisions for sentence, phrase, and rand conditions
    block_conditions = {
        'sentence': {
            'block1': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'block2': ['B6', 'B7', 'B8', 'B9', 'BA']
        },
        'phrase': {
            'block1': ['NP'],
            'block2': ['VP']
        },
        'rand': {
            'block1': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'block2': ['R6', 'R7', 'R8', 'R9', 'RA']
        }
    }

    # Determine which condition we are working with
    if 'sentence' in corpus_name.lower():
        blocks = block_conditions['sentence']
    elif 'phrase' in corpus_name.lower():
        blocks = block_conditions['phrase']
    elif 'rand' in corpus_name.lower():
        blocks = block_conditions['rand']
    else:
        raise ValueError("Unknown corpus name.")

    # Process each block separately
    for block_name, trials in blocks.items():
        # Prepare the HDF5 file for this block
        hdf5_path = os.path.join(output_dir, f"{model_name}_{corpus_name}_{block_name}_activations.hdf5")
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Filter the dataframe to only include rows belonging to the current block
            block_df = corpus_df[corpus_df.iloc[:, 0].str.startswith(tuple(trials))]

            for trial in tqdm(block_df.iloc[:, 0].unique(), desc=f"Processing {block_name} trials"):
                trial_df = block_df[block_df.iloc[:, 0] == trial]
                words = trial_df.iloc[:, 1].tolist()
                combined_text = ''.join(words)  # Assuming Chinese text

                # Tokenize and encode the text
                tokens = tokenizer.tokenize(combined_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_tensor = torch.tensor([input_ids]).to('cuda')

                mlp_activations = []

                # Hook function to capture MLP activations
                def hook_fn(module, input, output):
                    mlp_activations.append(output)

                # Register hooks for each MLP block
                hooks = []
                for block_index, block in enumerate(model.transformer.h):
                    hook = block.mlp.c_fc.register_forward_hook(hook_fn)
                    hooks.append(hook)

                # Run the model
                with torch.no_grad():
                    outputs = model(input_tensor, output_hidden_states=True)

                # Remove hooks after computation
                for hook in hooks:
                    hook.remove()

                # Check if activations were captured
                if not mlp_activations:
                    raise ValueError("No MLP activations were captured.")

                # Process activations for each layer and store them in HDF5
                for layer_index, layer_activations in enumerate(mlp_activations):
                    activations_layer = layer_activations.squeeze(0).cpu().numpy()

                    # Check for length mismatch between tokens and activations
                    if len(tokens) != activations_layer.shape[0]:
                        raise ValueError(f"Token mismatch: {len(tokens)} tokens, but {activations_layer.shape[0]} activations.")

                    # Only keep the last 32 activations (discard first 4)
                    if activations_layer.shape[0] < 36:
                        raise ValueError("Each trial should contain 36 characters.")
                    activations_trimmed = activations_layer[-32:, :]  # Keep the last 32 activations

                    # Store the activations in the HDF5 file
                    layer_group = hdf5_file.require_group(f'layer_{layer_index}')
                    trial_dataset_name = f'trial_{trial}'
                    if trial_dataset_name in layer_group:
                        del layer_group[trial_dataset_name]  # Overwrite if exists
                    layer_group.create_dataset(trial_dataset_name, data=activations_trimmed)

        print(f"Activations saved to {hdf5_path}")


def llama2_activation(corpus_path, model_name='Llama-2-7b-hf', output_dir='Results/correlation/activations'):
    """Extract neural activations from Llama2 model for the given corpus."""
    tokenizer = AutoTokenizer.from_pretrained(f"./models/Llama2/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"./models/Llama2/{model_name}", torch_dtype=torch.bfloat16, output_hidden_states=True).to('cuda')

    # Load the corpus
    corpus_df = pd.read_csv(corpus_path, header=None)
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # Define block divisions for sentence, phrase, and rand conditions
    block_conditions = {
        'sentence': {
            'block1': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'block2': ['B6', 'B7', 'B8', 'B9', 'BA']
        },
        'phrase': {
            'block1': ['NP'],
            'block2': ['VP']
        },
        'rand': {
            'block1': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'block2': ['R6', 'R7', 'R8', 'R9', 'RA']
        }
    }

    # Determine which condition we are working with
    if 'sentence' in corpus_name.lower():
        blocks = block_conditions['sentence']
    elif 'phrase' in corpus_name.lower():
        blocks = block_conditions['phrase']
    elif 'rand' in corpus_name.lower():
        blocks = block_conditions['rand']
    else:
        raise ValueError("Unknown corpus name.")

    # Process each block separately
    for block_name, trials in blocks.items():
        # Prepare the HDF5 file for this block
        hdf5_path = os.path.join(output_dir, f"{model_name}_{corpus_name}_{block_name}_activations.hdf5")
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Filter the dataframe to only include rows belonging to the current block
            block_df = corpus_df[corpus_df.iloc[:, 0].str.startswith(tuple(trials))]

            for trial in tqdm(block_df.iloc[:, 0].unique(), desc=f"Processing {block_name} trials"):
                trial_df = block_df[block_df.iloc[:, 0] == trial]
                words = trial_df.iloc[:, 1].tolist()
                combined_text = ''.join(words)  # Assuming Chinese text

                # Tokenize and encode the text
                tokens = tokenizer.tokenize(combined_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_tensor = torch.tensor([input_ids]).to('cuda')

                mlp_activations = []

                # Hook function to capture MLP activations
                def hook_fn(module, input, output):
                    gate_output = module.gate_proj(input[0])
                    up_output = module.up_proj(input[0])
                    activation_output = module.act_fn(gate_output) * up_output
                    mlp_activations.append(activation_output.to(torch.float32).detach().cpu().numpy())

                # Register hooks for each transformer block
                hooks = []
                for block in model.model.layers:
                    hook = block.mlp.register_forward_hook(hook_fn)
                    hooks.append(hook)

                # Run the model
                with torch.no_grad():
                    model(input_tensor, output_hidden_states=True)

                # Remove hooks after computation
                for hook in hooks:
                    hook.remove()

                # Check if activations were captured
                if not mlp_activations:
                    raise ValueError("No MLP activations were captured.")

                # Process activations for each layer and store them in HDF5
                for layer_index, layer_activations in enumerate(mlp_activations):
                    activations_layer = layer_activations.squeeze(0)

                    # Check for length mismatch between tokens and activations
                    if len(tokens) != activations_layer.shape[0]:
                        raise ValueError(f"Token mismatch: {len(tokens)} tokens, but {activations_layer.shape[0]} activations.")

                    # Only keep the last 32 activations (discard first 4)
                    if activations_layer.shape[0] < 36:
                        raise ValueError("Each trial should contain 36 characters.")
                    activations_trimmed = activations_layer[-32:, :]  # Keep the last 32 activations

                    # Store the activations in the HDF5 file
                    layer_group = hdf5_file.require_group(f'layer_{layer_index}')
                    trial_dataset_name = f'trial_{trial}'
                    if trial_dataset_name in layer_group:
                        del layer_group[trial_dataset_name]  # Overwrite if exists
                    layer_group.create_dataset(trial_dataset_name, data=activations_trimmed)

        print(f"Activations saved to {hdf5_path}")


def llama3_1_activation(corpus_path, model_name='Llama-3.1-8B', output_dir='Results/correlation/activations'):
    """Extract neural activations from Llama3.1 model for the given corpus."""
    tokenizer = AutoTokenizer.from_pretrained(f"./models/Llama3/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"./models/Llama3/{model_name}", torch_dtype=torch.float16,
                                                 device_map="auto", output_hidden_states=True)

    # Load the corpus
    corpus_df = pd.read_csv(corpus_path, header=None)
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # Define block divisions for sentence, phrase, and rand conditions
    block_conditions = {
        'sentence': {
            'block1': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'block2': ['B6', 'B7', 'B8', 'B9', 'BA']
        },
        'phrase': {
            'block1': ['NP'],
            'block2': ['VP']
        },
        'rand': {
            'block1': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'block2': ['R6', 'R7', 'R8', 'R9', 'RA']
        }
    }

    # Determine which condition we are working with
    if 'sentence' in corpus_name.lower():
        blocks = block_conditions['sentence']
    elif 'phrase' in corpus_name.lower():
        blocks = block_conditions['phrase']
    elif 'rand' in corpus_name.lower():
        blocks = block_conditions['rand']
    else:
        raise ValueError("Unknown corpus name.")

    # Process each block separately
    for block_name, trials in blocks.items():
        # Prepare the HDF5 file for this block
        hdf5_path = os.path.join(output_dir, f"{model_name}_{corpus_name}_{block_name}_activations.hdf5")
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Filter the dataframe to only include rows belonging to the current block
            block_df = corpus_df[corpus_df.iloc[:, 0].str.startswith(tuple(trials))]

            for trial in tqdm(block_df.iloc[:, 0].unique(), desc=f"Processing {block_name} trials"):
                trial_df = block_df[block_df.iloc[:, 0] == trial]
                words = trial_df.iloc[:, 1].tolist()
                combined_text = ''.join(words)  # Assuming Chinese text
                combined_text_with_separators = '|'.join(combined_text)

                tokens = tokenizer.tokenize(combined_text_with_separators)
                input_ids_with_separators = tokenizer.convert_tokens_to_ids(tokens)
                separator_token_id = tokenizer.convert_tokens_to_ids(['|'])[0]
                input_ids = []
                token_groups = []
                current_group = []

                # Group tokens by handling separators
                for token_id in input_ids_with_separators:
                    if token_id == separator_token_id:
                        if current_group:
                            token_groups.append(current_group)
                            current_group = []
                    else:
                        input_ids.append(token_id)
                        current_group.append(token_id)
                if current_group:
                    token_groups.append(current_group)

                input_tensor = torch.tensor([input_ids]).to('cuda')

                mlp_activations = []

                # Hook function to capture MLP activations
                def hook_fn(module, input, output):
                    gate_output = module.gate_proj(input[0])
                    up_output = module.up_proj(input[0])
                    activation_output = module.act_fn(gate_output) * up_output
                    mlp_activations.append(activation_output)

                # Register hooks for each transformer block
                hooks = []
                for block in model.model.layers:
                    hook = block.mlp.register_forward_hook(hook_fn)
                    hooks.append(hook)

                # Run the model
                with torch.no_grad():
                    model(input_ids=input_tensor, output_hidden_states=True)

                # Remove hooks after computation
                for hook in hooks:
                    hook.remove()

                # Check if activations were captured
                if not mlp_activations:
                    raise ValueError("No MLP activations were captured.")

                # Process activations for each layer and store them in HDF5
                for layer_index, layer_activations in enumerate(mlp_activations):
                    averaged_activations = []

                    for group in token_groups:
                        if len(group) > 1:
                            group_activations = layer_activations[:,
                                                len(averaged_activations):len(averaged_activations) + len(group)]
                            activation = torch.mean(group_activations, dim=1).cpu().numpy()
                        else:
                            activation = layer_activations[:, len(averaged_activations)].cpu().numpy()
                        averaged_activations.append(activation)

                    # Squeeze the activations to remove the extra batch dimension
                    averaged_activations = np.squeeze(np.array(averaged_activations))

                    # Check for length mismatch between tokens and activations
                    if len(combined_text) != averaged_activations.shape[0]:
                        raise ValueError(
                            f"Token mismatch: {len(tokens)} tokens, but {averaged_activations.shape[0]} activations.")

                    # Only keep the last 32 activations (discard first 4)
                    if averaged_activations.shape[0] < 36:
                        raise ValueError("Each trial should contain 36 characters.")
                    activations_trimmed = averaged_activations[-32:, :]  # Keep the last 32 activations

                    # Store the activations in the HDF5 file
                    layer_group = hdf5_file.require_group(f'layer_{layer_index}')
                    trial_dataset_name = f'trial_{trial}'
                    if trial_dataset_name in layer_group:
                        del layer_group[trial_dataset_name]  # Overwrite if exists
                    layer_group.create_dataset(trial_dataset_name, data=activations_trimmed)

        print(f"Activations saved to {hdf5_path}")


def gemma_activation(corpus_path, model_name='gemma-2b', output_dir='Results/correlation/activations'):
    """Extract neural activations from Gemma model for the given corpus."""
    tokenizer = AutoTokenizer.from_pretrained(f"models/gemma/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"models/gemma/{model_name}", torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=False)

    # Load the corpus
    corpus_df = pd.read_csv(corpus_path, header=None)
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # Define block divisions for sentence, phrase, and rand conditions
    block_conditions = {
        'sentence': {
            'block1': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'block2': ['B6', 'B7', 'B8', 'B9', 'BA']
        },
        'phrase': {
            'block1': ['NP'],
            'block2': ['VP']
        },
        'rand': {
            'block1': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'block2': ['R6', 'R7', 'R8', 'R9', 'RA']
        }
    }

    # Determine which condition we are working with
    if 'sentence' in corpus_name.lower():
        blocks = block_conditions['sentence']
    elif 'phrase' in corpus_name.lower():
        blocks = block_conditions['phrase']
    elif 'rand' in corpus_name.lower():
        blocks = block_conditions['rand']
    else:
        raise ValueError("Unknown corpus name.")

    # Process each block separately
    for block_name, trials in blocks.items():
        # Prepare the HDF5 file for this block
        hdf5_path = os.path.join(output_dir, f"{model_name}_{corpus_name}_{block_name}_activations.hdf5")
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Filter the dataframe to only include rows belonging to the current block
            block_df = corpus_df[corpus_df.iloc[:, 0].str.startswith(tuple(trials))]

            for trial in tqdm(block_df.iloc[:, 0].unique(), desc=f"Processing {block_name} trials"):
                trial_df = block_df[block_df.iloc[:, 0] == trial]
                words = trial_df.iloc[:, 1].tolist()
                combined_text = ''.join(words)  # Assuming Chinese text
                combined_text_with_separators = '|'.join(combined_text)

                tokens = tokenizer.tokenize(combined_text_with_separators)
                input_ids_with_separators = tokenizer.convert_tokens_to_ids(tokens)
                separator_token_id = tokenizer.convert_tokens_to_ids(['|'])[0]
                input_ids = []
                token_groups = []
                current_group = []

                # Group the tokens to handle separators
                for token_id in input_ids_with_separators:
                    if token_id == separator_token_id:
                        if current_group:
                            token_groups.append(current_group)
                            current_group = []
                    else:
                        input_ids.append(token_id)
                        current_group.append(token_id)
                if current_group:
                    token_groups.append(current_group)

                input_tensor = torch.tensor([input_ids]).to('cuda')

                mlp_activations = []

                # Hook function to capture MLP activations
                def hook_fn(module, input, output):
                    gate_output = module.gate_proj(input[0])
                    up_output = module.up_proj(input[0])
                    activation_output = module.act_fn(gate_output) * up_output
                    mlp_activations.append(activation_output)

                # Register hooks for each transformer block
                hooks = []
                for block in model.model.layers:
                    hook = block.mlp.register_forward_hook(hook_fn)
                    hooks.append(hook)

                # Run the model
                with torch.no_grad():
                    model(input_ids=input_tensor)

                # Remove hooks after computation
                for hook in hooks:
                    hook.remove()

                # Check if activations were captured
                if not mlp_activations:
                    raise ValueError("No MLP activations were captured.")

                for layer_index, layer_activations in enumerate(mlp_activations):
                    averaged_activations = []

                    for group in token_groups:
                        if len(group) > 1:
                            group_activations = layer_activations[:,
                                                len(averaged_activations):len(averaged_activations) + len(group)]
                            activation = torch.mean(group_activations, dim=1).cpu().numpy()
                        else:
                            activation = layer_activations[:, len(averaged_activations)].cpu().numpy()
                        averaged_activations.append(activation)

                    # Squeeze the activations to remove the extra batch dimension
                    averaged_activations = np.squeeze(np.array(averaged_activations))

                    # Check for length mismatch between tokens and activations
                    if len(combined_text) != averaged_activations.shape[0]:
                        raise ValueError(f"Token mismatch: {len(tokens)} tokens, but {averaged_activations.shape[0]} activations.")

                    # Only keep the last 32 activations (discard first 4)
                    if averaged_activations.shape[0] < 36:
                        raise ValueError("Each trial should contain 36 characters.")
                    activations_trimmed = averaged_activations[-32:, :]  # Keep the last 32 activations

                    # Store the activations in the HDF5 file
                    layer_group = hdf5_file.require_group(f'layer_{layer_index}')
                    trial_dataset_name = f'trial_{trial}'
                    if trial_dataset_name in layer_group:
                        del layer_group[trial_dataset_name]  # Overwrite if exists
                    layer_group.create_dataset(trial_dataset_name, data=activations_trimmed)

        print(f"Activations saved to {hdf5_path}")


def glm_activation(corpus_path, model_name='GLM-4-9b', output_dir='Results/correlation/activations'):
    """Extract neural activations from GLM model for the given corpus."""
    tokenizer = AutoTokenizer.from_pretrained(f"models/GLM/{model_name}", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        f"models/GLM/{model_name}",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda')

    # Load the corpus
    corpus_df = pd.read_csv(corpus_path, header=None)
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]

    # Define block divisions for sentence, phrase, and rand conditions
    block_conditions = {
        'sentence': {
            'block1': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'block2': ['B6', 'B7', 'B8', 'B9', 'BA']
        },
        'phrase': {
            'block1': ['NP'],
            'block2': ['VP']
        },
        'rand': {
            'block1': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'block2': ['R6', 'R7', 'R8', 'R9', 'RA']
        }
    }

    # Determine which condition we are working with
    if 'sentence' in corpus_name.lower():
        blocks = block_conditions['sentence']
    elif 'phrase' in corpus_name.lower():
        blocks = block_conditions['phrase']
    elif 'rand' in corpus_name.lower():
        blocks = block_conditions['rand']
    else:
        raise ValueError("Unknown corpus name.")

    # Process each block separately
    for block_name, trials in blocks.items():
        # Prepare the HDF5 file for this block
        hdf5_path = os.path.join(output_dir, f"{model_name}_{corpus_name}_{block_name}_activations.hdf5")
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Filter the dataframe to only include rows belonging to the current block
            block_df = corpus_df[corpus_df.iloc[:, 0].str.startswith(tuple(trials))]

            for trial in tqdm(block_df.iloc[:, 0].unique(), desc=f"Processing {block_name} trials"):
                trial_df = block_df[block_df.iloc[:, 0] == trial]
                words = trial_df.iloc[:, 1].tolist()
                combined_text = ''.join(words)  # Assuming Chinese text
                combined_text_with_separators = '|'.join(combined_text)

                tokens = tokenizer.tokenize(combined_text_with_separators)
                input_ids_with_separators = tokenizer.convert_tokens_to_ids(tokens)
                separator_token_id = 91  # Separator ID specific to GLM
                input_ids = []
                token_groups = []
                current_group = []

                # Group the tokens to handle separators
                for token_id in input_ids_with_separators:
                    if token_id == separator_token_id:
                        if current_group:
                            token_groups.append(current_group)
                            current_group = []
                    else:
                        input_ids.append(token_id)
                        current_group.append(token_id)
                if current_group:
                    token_groups.append(current_group)

                input_tensor = torch.tensor([input_ids]).to('cuda')

                mlp_activations = []

                # Hook function to capture MLP activations
                def hook_fn(module, input, output):
                    mlp_activations.append(input[0].float())

                # Register hooks for each transformer block
                hooks = []
                for block in model.transformer.encoder.layers:
                    hook = block.mlp.dense_4h_to_h.register_forward_hook(hook_fn)
                    hooks.append(hook)

                # Run the model
                with torch.no_grad():
                    model(input_ids=input_tensor)

                # Remove hooks after computation
                for hook in hooks:
                    hook.remove()

                # Check if activations were captured
                if not mlp_activations:
                    raise ValueError("No MLP activations were captured.")

                for layer_index, layer_activations in enumerate(mlp_activations):
                    averaged_activations = []

                    for group in token_groups:
                        if len(group) > 1:
                            group_activations = layer_activations[:,
                                                len(averaged_activations):len(averaged_activations) + len(group)]
                            activation = torch.mean(group_activations, dim=1).cpu().numpy()
                        else:
                            activation = layer_activations[:, len(averaged_activations)].cpu().numpy()
                        averaged_activations.append(activation)

                    # Squeeze the activations to remove the extra batch dimension
                    averaged_activations = np.squeeze(np.array(averaged_activations))

                    # Check for length mismatch between tokens and activations
                    if len(combined_text) != averaged_activations.shape[0]:
                        raise ValueError(f"Token mismatch: {len(tokens)} tokens, but {averaged_activations.shape[0]} activations.")

                    # Only keep the last 32 activations (discard first 4)
                    if averaged_activations.shape[0] < 36:
                        raise ValueError("Each trial should contain 36 characters.")
                    activations_trimmed = averaged_activations[-32:, :]  # Keep the last 32 activations

                    # Store the activations in the HDF5 file
                    layer_group = hdf5_file.require_group(f'layer_{layer_index}')
                    trial_dataset_name = f'trial_{trial}'
                    if trial_dataset_name in layer_group:
                        del layer_group[trial_dataset_name]  # Overwrite if exists
                    layer_group.create_dataset(trial_dataset_name, data=activations_trimmed)

        print(f"Activations saved to {hdf5_path}")


def permutation_test(model_name, corpus_name, input_dir='Results/correlation/activations',
                     output_dir='Results/correlation/permutations'):
    """Perform permutation testing on neural activations to identify significant frequencies."""
    blocks = ['block1', 'block2']  # Process both experimental blocks

    for block in blocks:
        # Load the activations from HDF5 (now block-specific)
        input_path = os.path.join(input_dir, f"{model_name}_{corpus_name}_{block}_activations.hdf5")
        hdf5_file = h5py.File(input_path, 'r')

        layers = list(hdf5_file.keys())

        # Prepare the HDF5 file for storing permutation significant neurons
        output_path = os.path.join(output_dir, f"permutations/{model_name}_{corpus_name}_{block}_permutation.hdf5")
        with h5py.File(output_path, 'w') as output_file:
            for layer in tqdm(layers, desc=f"Permutation testing for {block}"):
                layer_data = []
                for trial in hdf5_file[layer]:
                    trial_data = hdf5_file[layer][trial][:]
                    layer_data.append(trial_data)
                layer_data = np.stack(layer_data, axis=0)  # Shape: (num_trials, num_tokens, num_neurons)

                num_trials, num_tokens, num_neurons = layer_data.shape

                # Perform permutation test for each neuron
                for neuron_idx in range(num_neurons):
                    neuron_data = layer_data[:, :, neuron_idx].reshape(-1)
                    real_fft = fft(neuron_data)
                    real_1hz = np.real(real_fft[1])
                    real_2hz = np.real(real_fft[2])

                    # Perform permutations
                    perm_1hz = []
                    perm_2hz = []
                    for _ in range(1000):
                        perm_data = np.random.permutation(neuron_data)
                        perm_fft = fft(perm_data)
                        perm_1hz.append(np.real(perm_fft[1]))
                        perm_2hz.append(np.real(perm_fft[2]))

                    # Convert lists to numpy arrays
                    perm_1hz = np.array(perm_1hz)
                    perm_2hz = np.array(perm_2hz)

                    # Store the permutation results
                    layer_group = output_file.require_group(layer)
                    neuron_dataset_name_real = f'neuron_{neuron_idx}_real'
                    neuron_dataset_name_perm_1hz = f'neuron_{neuron_idx}_perm_1hz'
                    neuron_dataset_name_perm_2hz = f'neuron_{neuron_idx}_perm_2hz'
                    if neuron_dataset_name_real in layer_group:
                        del layer_group[neuron_dataset_name_real]  # Overwrite if exists
                    if neuron_dataset_name_perm_1hz in layer_group:
                        del layer_group[neuron_dataset_name_perm_1hz]  # Overwrite if exists
                    if neuron_dataset_name_perm_2hz in layer_group:
                        del layer_group[neuron_dataset_name_perm_2hz]  # Overwrite if exists
                    layer_group.create_dataset(neuron_dataset_name_real, data=[real_1hz, real_2hz])
                    layer_group.create_dataset(neuron_dataset_name_perm_1hz, data=perm_1hz)
                    layer_group.create_dataset(neuron_dataset_name_perm_2hz, data=perm_2hz)

        print(f"Permutation test for {block} saved to {output_path}")


def significant_neurons_zscore(model_name, corpus_name, confidence_interval, activation_data_dir, output_dir):
    """Identify significant neurons using z-score analysis after permutation testing."""
    z_threshold = 2.0  # Z-score threshold for significance
    blocks = ['block1', 'block2']  # Blocks to process

    # Initialize lists to store dataframes for each block
    permutation_dfs = []
    significant_neurons_dfs = []

    num_layers = None
    num_neurons = None

    # Process each block separately
    for block in blocks:
        # Load the permutation test significant neurons (block-specific)
        permutation_path = f'{output_dir}/permutations/{model_name}_{corpus_name}_{block}_permutation.hdf5'
        hdf5_file = h5py.File(permutation_path, 'r')

        layers = sorted(list(hdf5_file.keys()), key=lambda x: int(x.split('_')[1]))  # Sort layers by index
        num_layers = len(layers) if num_layers is None else num_layers
        # Determine the number of neurons in each layer
        num_neurons = len([key for key in hdf5_file[layers[0]].keys() if key.startswith('neuron_') and key.endswith('_real')]) if num_neurons is None else num_neurons

        significant_permutation_neurons = {
            'Layer': [],
            'significant_si_neurons': [],
            'number_of_si_neurons': [],
            'significant_pi_neurons': [],
            'number_of_pi_neurons': [],
            'shared_neurons': [],
            'number_of_shared_neurons': []
        }

        for layer in tqdm(layers, desc=f"Processing significant permutation neurons for {block}"):
            layer_index = int(layer.split('_')[1]) + 1  # Adjust layer index to start from 1
            si_neurons = []
            pi_neurons = []
            for neuron in range(num_neurons):
                neuron_name_real = f'neuron_{neuron}_real'
                neuron_name_perm_1hz = f'neuron_{neuron}_perm_1hz'
                neuron_name_perm_2hz = f'neuron_{neuron}_perm_2hz'

                if neuron_name_real not in hdf5_file[layer]:
                    print(f"Warning: {neuron_name_real} not found in {layer}. Skipping this neuron.")
                    continue

                real_1hz, real_2hz = hdf5_file[layer][neuron_name_real][:]
                perm_1hz = hdf5_file[layer][neuron_name_perm_1hz][:]
                perm_2hz = hdf5_file[layer][neuron_name_perm_2hz][:]

                # Calculate one-sided confidence intervals
                ci_1hz = np.percentile(perm_1hz, confidence_interval)
                ci_2hz = np.percentile(perm_2hz, confidence_interval)

                si_neuron = real_1hz > ci_1hz
                pi_neuron = real_2hz > ci_2hz

                if si_neuron:
                    si_neurons.append(neuron)
                if pi_neuron:
                    pi_neurons.append(neuron)

            shared_neurons = list(set(si_neurons) & set(pi_neurons))

            significant_permutation_neurons['Layer'].append(layer_index)
            significant_permutation_neurons['significant_si_neurons'].append(sorted(si_neurons))
            significant_permutation_neurons['number_of_si_neurons'].append(len(si_neurons))
            significant_permutation_neurons['significant_pi_neurons'].append(sorted(pi_neurons))
            significant_permutation_neurons['number_of_pi_neurons'].append(len(pi_neurons))
            significant_permutation_neurons['shared_neurons'].append(sorted(shared_neurons))
            significant_permutation_neurons['number_of_shared_neurons'].append(len(shared_neurons))

        hdf5_file.close()

        # Create DataFrame for this block
        significant_permutation_df = pd.DataFrame(significant_permutation_neurons)
        significant_permutation_df['significant_si_neurons'] = significant_permutation_df['significant_si_neurons'].apply(lambda x: str(x))
        significant_permutation_df['significant_pi_neurons'] = significant_permutation_df['significant_pi_neurons'].apply(lambda x: str(x))
        significant_permutation_df['shared_neurons'] = significant_permutation_df['shared_neurons'].apply(lambda x: str(x))

        permutation_dfs.append(significant_permutation_df)

        # Now process activations and compute significant neurons per block
        # Load the activations from HDF5 for the experiment and control conditions
        exp_hdf5_file = h5py.File(f'{activation_data_dir}/{model_name}_{corpus_name}_{block}_activations.hdf5', 'r')
        ctrl_hdf5_file = h5py.File(f'{activation_data_dir}/{model_name}_SEEG_Rand_{block}_activations.hdf5', 'r')

        si_matrix = np.zeros((num_layers, num_neurons))
        pi_matrix = np.zeros((num_layers, num_neurons))

        for layer_index in tqdm(range(num_layers), desc=f"Processing significant zscore neurons for {block}"):
            layer_name = f'layer_{layer_index}'
            significant_si_neurons_block = set(ast.literal_eval(significant_permutation_df['significant_si_neurons'][layer_index]))
            significant_pi_neurons_block = set(ast.literal_eval(significant_permutation_df['significant_pi_neurons'][layer_index]))

            significant_neurons_set = significant_si_neurons_block | significant_pi_neurons_block

            if not significant_neurons_set:
                continue

            for neuron_index in significant_neurons_set:
                exp_activations = []
                ctrl_activations = []

                for trial in exp_hdf5_file[layer_name].keys():
                    trial_data = exp_hdf5_file[layer_name][trial][:]
                    if neuron_index >= trial_data.shape[1]:
                        print(f"Warning: neuron_index {neuron_index} out of bounds for trial {trial} in layer {layer_name}")
                        continue
                    exp_activations.append(trial_data[:, neuron_index])

                for trial in ctrl_hdf5_file[layer_name].keys():
                    trial_data = ctrl_hdf5_file[layer_name][trial][:]
                    if neuron_index >= trial_data.shape[1]:
                        print(f"Warning: neuron_index {neuron_index} out of bounds for trial {trial} in layer {layer_name}")
                        continue
                    ctrl_activations.append(trial_data[:, neuron_index])

                if not exp_activations or not ctrl_activations:
                    continue

                exp_activations = np.concatenate(exp_activations)
                ctrl_activations = np.concatenate(ctrl_activations)

                exp_fft = rfft(exp_activations)
                ctrl_fft = rfft(ctrl_activations)
                exp_fft[0] = ctrl_fft[0] = 0  # Zero out the DC component

                freqs = rfftfreq(len(exp_activations), d=1. / 4)  # Assuming 1Hz sampling rate

                idx_1hz = np.argmin(np.abs(freqs - 1.0))
                idx_2hz = np.argmin(np.abs(freqs - 2.0))

                si = np.abs(exp_fft[idx_1hz]) - np.abs(ctrl_fft[idx_1hz])
                pi = np.abs(exp_fft[idx_2hz]) - np.abs(ctrl_fft[idx_2hz])

                if neuron_index in significant_si_neurons_block:
                    si_matrix[layer_index, neuron_index] = si
                if neuron_index in significant_pi_neurons_block:
                    pi_matrix[layer_index, neuron_index] = pi

        exp_hdf5_file.close()
        ctrl_hdf5_file.close()

        # Calculate the mean and standard deviation for SI and PI
        si_mean = np.mean(si_matrix)
        si_std = np.std(si_matrix)
        pi_mean = np.mean(pi_matrix)
        pi_std = np.std(pi_matrix)

        # Convert SI and PI to z-scores
        si_zscores = (si_matrix - si_mean) / si_std
        pi_zscores = (pi_matrix - pi_mean) / pi_std

        significant_si_counts = np.zeros(num_layers)
        significant_pi_counts = np.zeros(num_layers)
        significant_si_neurons_list = []
        significant_pi_neurons_list = []
        shared_neurons_list = []
        shared_counts = []

        # Identify significant neurons based on z-scores
        for layer_index in range(num_layers):
            significant_si_indices = np.where(si_zscores[layer_index, :] > z_threshold)[0]
            significant_si_indices = sorted(significant_si_indices, key=lambda i: -si_matrix[layer_index, i])

            significant_pi_indices = np.where(pi_zscores[layer_index, :] > z_threshold)[0]
            significant_pi_indices = sorted(significant_pi_indices, key=lambda i: -pi_matrix[layer_index, i])

            significant_si_counts[layer_index] = len(significant_si_indices)
            significant_pi_counts[layer_index] = len(significant_pi_indices)

            significant_si_neurons_list.append(significant_si_indices)
            significant_pi_neurons_list.append(significant_pi_indices)
            shared_neurons_layer = list(set(significant_si_indices) & set(significant_pi_indices))
            shared_neurons_list.append(shared_neurons_layer)
            shared_counts.append(len(shared_neurons_layer))

        # Create DataFrame for this block
        significant_neurons_df = pd.DataFrame({
            'Layer': [i + 1 for i in range(num_layers)],
            'significant_si_neurons': [f"[{', '.join(map(str, sorted(neurons)))}]" for neurons in significant_si_neurons_list],
            'number_of_si_neurons': significant_si_counts,
            'significant_pi_neurons': [f"[{', '.join(map(str, sorted(neurons)))}]" for neurons in significant_pi_neurons_list],
            'number_of_pi_neurons': significant_pi_counts,
            'shared_neurons': [f"[{', '.join(map(str, sorted(neurons)))}]" for neurons in shared_neurons_list],
            'number_of_shared_neurons': shared_counts
        })

        # Sort the DataFrame by Layer
        significant_neurons_df = significant_neurons_df.sort_values(by='Layer')

        significant_neurons_dfs.append(significant_neurons_df)

    # After processing both blocks, merge the dataframes

    # Merge significant_permutation_df dataframes
    merged_permutation_df = permutation_dfs[0].copy()
    for df in permutation_dfs[1:]:
        for idx in merged_permutation_df.index:
            # Merge significant_si_neurons
            neurons1 = set(ast.literal_eval(merged_permutation_df.at[idx, 'significant_si_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'significant_si_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_permutation_df.at[idx, 'significant_si_neurons'] = str(merged_neurons)
            merged_permutation_df.at[idx, 'number_of_si_neurons'] = len(merged_neurons)

            # Merge significant_pi_neurons
            neurons1 = set(ast.literal_eval(merged_permutation_df.at[idx, 'significant_pi_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'significant_pi_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_permutation_df.at[idx, 'significant_pi_neurons'] = str(merged_neurons)
            merged_permutation_df.at[idx, 'number_of_pi_neurons'] = len(merged_neurons)

            # Merge shared_neurons
            neurons1 = set(ast.literal_eval(merged_permutation_df.at[idx, 'shared_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'shared_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_permutation_df.at[idx, 'shared_neurons'] = str(merged_neurons)
            merged_permutation_df.at[idx, 'number_of_shared_neurons'] = len(merged_neurons)

    # Save the merged permutation dataframe
    perm_output_path = f'{output_dir}/significant_neurons/{model_name}_{corpus_name}_permutation_neurons.csv'
    merged_permutation_df.to_csv(perm_output_path, index=False)
    print(f"Merged significant permutation neurons saved to {perm_output_path}")

    # Merge significant_neurons_df dataframes
    merged_significant_df = significant_neurons_dfs[0].copy()
    for df in significant_neurons_dfs[1:]:
        for idx in merged_significant_df.index:
            # Merge significant_si_neurons
            neurons1 = set(ast.literal_eval(merged_significant_df.at[idx, 'significant_si_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'significant_si_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_significant_df.at[idx, 'significant_si_neurons'] = f"[{', '.join(map(str, merged_neurons))}]"
            merged_significant_df.at[idx, 'number_of_si_neurons'] = len(merged_neurons)

            # Merge significant_pi_neurons
            neurons1 = set(ast.literal_eval(merged_significant_df.at[idx, 'significant_pi_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'significant_pi_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_significant_df.at[idx, 'significant_pi_neurons'] = f"[{', '.join(map(str, merged_neurons))}]"
            merged_significant_df.at[idx, 'number_of_pi_neurons'] = len(merged_neurons)

            # Merge shared_neurons
            neurons1 = set(ast.literal_eval(merged_significant_df.at[idx, 'shared_neurons']))
            neurons2 = set(ast.literal_eval(df.at[idx, 'shared_neurons']))
            merged_neurons = sorted(neurons1.union(neurons2))
            merged_significant_df.at[idx, 'shared_neurons'] = f"[{', '.join(map(str, merged_neurons))}]"
            merged_significant_df.at[idx, 'number_of_shared_neurons'] = len(merged_neurons)

    # Save the merged significant neurons dataframe
    output_path = f'{output_dir}/significant_neurons/{model_name}_{corpus_name}_significant_neurons.csv'
    merged_significant_df.to_csv(output_path, index=False)
    print(f"Merged significant neurons saved to {output_path}")


def calculate_correlation(model_name, corpus_name, conditions, activation_data_dir, output_dir, random_effect=False):
    """Calculate correlations between model representations and sEEG data."""
    # Step 1: Processing Significant Neurons (LLM Side)
    def extract_significant_neurons(corpus_name, neuron_type, random_effect=False):
        """
        Extract significant neurons or generate random neurons with the same count.
        For random effect, excludes neurons in the permutation file.

        Args:
            corpus_name (str): Name of the corpus
            neuron_type (str): Type of neurons ('si', 'pi', or 'shared')
            random_effect (bool): Whether to generate random neurons instead of using significant neurons

        Returns:
            list: List of neuron indices for each layer
        """
        significant_neurons_file = f'Results/correlation/significant_neurons_zscore/{model_name}_{corpus_name}_significant_neurons.csv'
        neurons_df = pd.read_csv(significant_neurons_file)

        if neuron_type == 'si':
            significant_neurons = neurons_df['exclusive_si_neurons'].apply(eval).values
        elif neuron_type == 'pi':
            significant_neurons = neurons_df['exclusive_pi_neurons'].apply(eval).values
        elif neuron_type == 'shared':
            significant_neurons = neurons_df['shared_neurons'].apply(eval).values
        else:
            return None

        if not random_effect:
            return significant_neurons

        # For random effect, we need to exclude neurons from the permutation file
        permutation_file = f'Results/correlation/significant_neurons_zscore/{model_name}_{corpus_name}_permutation_neurons.csv'
        if os.path.exists(permutation_file):
            perm_df = pd.read_csv(permutation_file)
            # Extract neurons to exclude based on neuron_type
            if neuron_type == 'si':
                neurons_to_exclude = perm_df['exclusive_si_neurons'].apply(eval).values
            elif neuron_type == 'pi':
                neurons_to_exclude = perm_df['exclusive_pi_neurons'].apply(eval).values
            elif neuron_type == 'shared':
                neurons_to_exclude = perm_df['shared_neurons'].apply(eval).values
            else:
                neurons_to_exclude = [[] for _ in range(len(significant_neurons))]
        else:
            # If permutation file doesn't exist, use empty exclusion lists
            neurons_to_exclude = [[] for _ in range(len(significant_neurons))]

        # Generate random neurons with the same count as significant neurons
        np.random.seed(42)  # Set seed for reproducibility
        random_neurons = []

        # Find a valid sample file to get neuron counts
        if 'Sentence' in corpus_name:
            block_name = 'Sentence_block1'
        elif 'Phrase' in corpus_name:
            block_name = 'Phrase_block1'
        else:
            block_name = 'Rand_block1'

        sample_file = f'{activation_data_dir}/{model_name}_SEEG_{block_name}_activations.hdf5'

        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"No activation file found for model {model_name}")

        # Get the number of neurons per layer
        neuron_counts = []
        with h5py.File(sample_file, 'r') as f:
            for layer_idx in range(len(significant_neurons)):
                layer_key = f'layer_{layer_idx}'
                if layer_key in f:
                    # Get sample trial data to determine neuron count
                    trial_keys = list(f[layer_key].keys())
                    if trial_keys:
                        trial_data = f[layer_key][trial_keys[0]][:]
                        neuron_counts.append(trial_data.shape[1])
                    else:
                        neuron_counts.append(0)
                else:
                    neuron_counts.append(0)

        # Generate random neurons for each layer
        for layer_idx, sig_neurons in enumerate(significant_neurons):
            num_neurons = len(sig_neurons)
            if neuron_counts[layer_idx] > 0 and num_neurons > 0:
                # Create a pool of valid neurons by excluding the permutation neurons
                exclude_set = set(neurons_to_exclude[layer_idx])
                all_neurons = set(range(neuron_counts[layer_idx]))
                valid_neurons = list(all_neurons - exclude_set)

                # If there aren't enough valid neurons, log a warning and allow replacements
                if len(valid_neurons) < num_neurons:
                    print(
                        f"Warning: Not enough valid neurons for {model_name}, {corpus_name}, layer {layer_idx}. Using replacements.")
                    # Use the valid neurons we have with replacement
                    random_layer_neurons = np.random.choice(valid_neurons, size=num_neurons, replace=True).tolist()
                else:
                    # Randomly select neurons without replacement (since we have enough)
                    random_layer_neurons = np.random.choice(valid_neurons, size=num_neurons, replace=False).tolist()

                random_neurons.append(random_layer_neurons)
            else:
                random_neurons.append([])

        return random_neurons

    def compute_rdm_for_neurons(neuron_indices, courpus_name, block_names):
        rdm_list = []  # This will store RDMs for each layer

        # Loop over each layer and its significant neurons
        for layer_idx, neuron_list in enumerate(neuron_indices):
            layer_rdm = []  # Store RDMs for neurons in the current layer

            # Loop over each significant neuron in the layer
            for neuron_idx in neuron_list:
                activations = []  # This will hold activations from each block for this neuron

                # Extract activations for each block
                for block_name in block_names:
                    # Path to the HDF5 file for the block's activations
                    hdf5_file = f'{activation_data_dir}/{model_name}_SEEG_{block_name}_activations.hdf5'

                    with h5py.File(hdf5_file, 'r') as f:
                        # Access the layer group in the HDF5 file
                        layer_group = f[f'layer_{layer_idx}']

                        # Collect activations for each trial
                        block_activations = []
                        for trial in layer_group.keys():
                            trial_data = layer_group[trial][:]
                            block_activations.append(trial_data[:, neuron_idx])  # Get neuron activations for this trial

                        # Compute the mean activation across all trials for the block
                        mean_block_activation = np.mean(block_activations, axis=0)
                        activations.append(mean_block_activation)  # Append the block's mean activation

                # Perform FFT on activations for all blocks
                rfft_activations = [rfft(a) for a in activations]

                # Define the frequency range (0.5 - 2 Hz)
                freqs = rfftfreq(len(activations[0]), d=1. / 4)  # 4 Hz sampling rate
                freq_mask = (freqs >= 0.5) & (freqs <= 2)

                # Apply frequency filter and take absolute value of the FFT
                activations_proj = [np.abs(a[freq_mask]) for a in rfft_activations]
                # Initialize an empty 6x6 RDM matrix
                rdm = np.zeros((6, 6))
                # Compute cosine similarity between all pairs of block activations
                for i in range(6):
                    for j in range(6):
                        rdm[i, j] = cosine(activations_proj[i], activations_proj[j])
                # Append the RDM matrix for this neuron
                layer_rdm.append(rdm)
            # Append the RDMs for the current layer
            rdm_list.append(layer_rdm)

        return rdm_list

    def average_rdm(rdm_list):
        avg_rdm_per_layer = []
        for layer_rdms in rdm_list:
            avg_rdm = np.mean(np.array(layer_rdms), axis=0)
            avg_rdm_per_layer.append(avg_rdm)
        return avg_rdm_per_layer

    # Step 2: Processing SEEG Data (SEEG Side)
    condition_map = {
        'SEEG_Sentence': 'sentence',
        'SEEG_Phrase': 'phrase',
        'SEEG_Rand': 'rand'
    }
    channels_file = f'SEEG_itpc/{condition_map[corpus_name]}_significant_channel_L.csv'
    channels_df = pd.read_csv(channels_file)

    def extract_itpc_data(sub_id, hemisphere, channel_idx, condition):
        itpc_data = []
        condition_block_map = {
            'sentence': ['Sentence_block1', 'Sentence_block2'],
            'phrase': ['Phrase_block1', 'Phrase_block2'],
            'rand': ['Rand_block1', 'Rand_block2']
        }

        # Loop through the blocks for the condition
        for block_name in condition_block_map[condition]:
            hdf5_file = f'Results/correlation/permutations/itpc_{hemisphere}.hdf5'
            channel_key = f"sub_{sub_id}_channel_{channel_idx}"

            with h5py.File(hdf5_file, 'r') as f:
                if channel_key not in f:
                    raise ValueError(f"Channel {channel_idx} for sub {sub_id} not found in file.")

                # Load ITPC results for the current block
                itpc_results = f[channel_key][block_name][:]
                itpc_data.append(itpc_results)

        return itpc_data

    def compute_rdm_for_seeg(sub_id, hemisphere):
        seeg_rdm_list = {}  # Dictionary to store RDM for each channel

        condition_block_names = ['Sentence_block1', 'Sentence_block2', 'Phrase_block1', 'Phrase_block2',
                                     'Rand_block1', 'Rand_block2']

        # Open the HDF5 file for the corresponding hemisphere
        hdf5_file = f'Results/correlation/permutations/itpc_{hemisphere}.hdf5'

        with h5py.File(hdf5_file, 'r') as f:
            # Loop through each channel for the current subject
            for channel_key in f.keys():
                if not channel_key.startswith(f"sub{sub_id}_"):
                    continue  # Skip if the channel does not belong to the current subject

                itpc_proj = []

                # Extract ITPC data for all 6 blocks (Sentence_block1, Sentence_block2, etc.)
                for block_name in condition_block_names:
                    itpc_results = f[channel_key][block_name][:]

                    # Apply FFT and filter by frequency range 0.5-2Hz
                    n_samples = (itpc_results.shape[0] - 1) * 2  # Total samples
                    freqs = np.fft.fftfreq(n_samples, d=1. / 512)[:itpc_results.shape[0]]
                    freq_mask = (freqs >= 0.5) & (freqs <= 2)

                    # Filter and project ITPC data onto the 0.5-2Hz range
                    itpc_proj_block = itpc_results[freq_mask]
                    itpc_proj.append(itpc_proj_block)

                # Now, compute the cosine similarity RDM between these 6 blocks
                rdm = np.zeros((6, 6))  # Initialize a 6x6 matrix

                for i in range(6):
                    for j in range(6):
                        rdm[i, j] = cosine(itpc_proj[i].flatten(), itpc_proj[j].flatten())  # Compute cosine similarity

                # Add the RDM for this channel to the seeg_rdm_list
                seeg_rdm_list[channel_key] = rdm

        return seeg_rdm_list

    # Step 3: Correlating RDMs between LLM and SEEG
    def correlate_rdms(rdm_model, seeg_rdm_list, itpc_file):
        """
        Correlate model RDM with SEEG RDMs.
        Args:
            rdm_model (numpy.ndarray): Model's RDM for a given layer.
            seeg_rdm_list (dict): SEEG RDMs for all channels.
            itpc_file (str): Path to the HDF5 file containing ITPC data.
        Returns:
            tuple: (sorted_correlations, top_channels_info)
        """
        # Check if the model RDM is valid
        if rdm_model is None or np.isnan(rdm_model).all():
            return [], []  # Return empty results if the model RDM is invalid

        correlation_results = []
        top_channels_info = []  # To store top 100 channels with their region labels

        with h5py.File(itpc_file, 'r') as f:
            for channel_key, rdm_seeg_channel in seeg_rdm_list.items():
                # Remove diagonal values before calculating Spearman correlation
                rdm_model_no_diag = rdm_model[~np.eye(rdm_model.shape[0], dtype=bool)]  # Remove diagonal
                rdm_seeg_no_diag = rdm_seeg_channel[~np.eye(rdm_seeg_channel.shape[0], dtype=bool)]  # Remove diagonal

                # Compute Spearman correlation between model RDM and SEEG RDM
                spearman_corr, _ = spearmanr(rdm_model_no_diag, rdm_seeg_no_diag, nan_policy='omit')  # Handle NaN
                spearman_corr = float(spearman_corr)  # Ensure the correlation is a float

                if np.isnan(spearman_corr):
                    spearman_corr = 0.0  # Set to 0 if NaN (in case of constant arrays)

                correlation_results.append((channel_key, spearman_corr))  # Store channel and correlation

                # Access `region_label` attribute from the HDF5 file
                if channel_key in f:
                    region_label = f[channel_key].attrs['region_label']
                    top_channels_info.append(
                        (channel_key, spearman_corr, region_label))  # Store channel, correlation, region
                else:
                    print(f"Warning: Channel {channel_key} not found in file.")

        # Sort by Spearman correlation and select ALL channels
        sorted_correlations = sorted(correlation_results, key=lambda x: x[1], reverse=True)
        top_channels_info = sorted(top_channels_info, key=lambda x: x[1], reverse=True)

        return sorted_correlations, top_channels_info

    def save_correlation_results(correlation_results, top_channels_info, hemisphere, model_name, corpus_name,
                                 output_dir, layer_idx, itpc_file, neuron_type, random_effect=False):
        if len(correlation_results) == 0:  # If no correlation results, skip saving
            return

        if random_effect:
            search_light_dir = f'{output_dir}/search_light_random'
        else:
            search_light_dir = f'{output_dir}/search_light'
        if not os.path.exists(search_light_dir):
            os.makedirs(search_light_dir)


        search_light_csv = f'{search_light_dir}/{model_name}_{corpus_name}_{hemisphere}_{neuron_type}_search_light.csv'
        spearman_csv = f'{search_light_dir}/{model_name}_{corpus_name}_{hemisphere}_{neuron_type}_spearman.csv'

        # Prepare data for search light ratios and region distribution
        region_count = {}  # Dictionary to hold the region counts for distribution
        region_correlation_sum = {}  # To accumulate correlation values per region for the search light ratio

        # For search_light.csv: We process all channels in `top_channels_info`
        for channel_key, correlation, region_label in top_channels_info:
            # Count occurrences of region labels for all channels (for search light calculation)
            if region_label not in region_count:
                region_count[region_label] = 0
                region_correlation_sum[region_label] = 0
            region_count[region_label] += 1
            region_correlation_sum[region_label] += correlation

        # Calculate search light ratios for each region (using all channels)
        search_light_ratios = {region: region_correlation_sum[region] / region_count[region]
                               for region in region_count}

        # Save search light ratios (layer-wise) – this uses ALL channels (not just top 100)
        search_light_df = pd.DataFrame([search_light_ratios], index=[f'layer_{layer_idx + 1}'])
        search_light_df.to_csv(search_light_csv, mode='a', header=not os.path.exists(search_light_csv))

        # For spearman.csv: No changes here, as we continue using all channels
        if os.path.exists(spearman_csv):
            # Load existing data
            spearman_df = pd.read_csv(spearman_csv)

            # Create new columns for the current layer
            spearman_data = {
                f'Channel_layer_{layer_idx + 1}': [channel_key for channel_key, _, _ in top_channels_info],
                f'Channel_region_label_{layer_idx + 1}': [region_label for _, _, region_label in top_channels_info],
                f'Spearman_layer_{layer_idx + 1}': [round(corr, 4) for _, corr, _ in top_channels_info]
            }

            # Append new layer columns to the existing DataFrame
            new_spearman_df = pd.DataFrame(spearman_data)
            spearman_df = pd.concat([spearman_df, new_spearman_df], axis=1)  # Concatenate column-wise
        else:
            # If the file doesn't exist, create new columns for the first layer
            spearman_data = {
                f'Channel_layer_{layer_idx + 1}': [channel_key for channel_key, _, _ in top_channels_info],
                f'Channel_region_label_{layer_idx + 1}': [region_label for _, _, region_label in top_channels_info],
                f'Spearman_layer_{layer_idx + 1}': [round(corr, 4) for _, corr, _ in top_channels_info]
            }
            spearman_df = pd.DataFrame(spearman_data)

        # Save the updated spearman DataFrame back to the CSV
        spearman_df.to_csv(spearman_csv, index=False)

        # For distribution.csv: We process ONLY the top 100 channels for each layer
        top_100_channels = top_channels_info[:100]  # Extract the top 100 channels based on correlation
        region_top_channel_count = {}  # To accumulate the count of top channels per region for distribution

        for channel_key, correlation, region_label in top_100_channels:
            # Count occurrences of region labels for the top 100 channels
            if region_label not in region_top_channel_count:
                region_top_channel_count[region_label] = 0
            region_top_channel_count[region_label] += 1


    # Step 4: Statistical Testing and Chi-Square Test
    # Implemented in the following codes

    # Step 5: Compute average top 10 Spearman correlations
    def compute_top_100_spearman(layer_spearman):
        """
        Compute the average of top 100 Spearman correlations:
        1. Compute top 100 average for each layer.
        2. Take the mean across all layers.
        """
        if not layer_spearman or len(layer_spearman) == 0:
            return float('nan')  # Handle empty input
        layer_means = []
        for layer_corr in layer_spearman:  # Iterate over layers
            if not layer_corr:
                continue  # Skip empty layers
            # Sort Spearman correlations for this layer
            sorted_corr = sorted([float(corr[1]) for corr in layer_corr], reverse=True)
            top_100 = sorted_corr[:100]  # Top 100 correlations for this layer
            if top_100:  # Check if there are valid correlations
                layer_means.append(np.mean(top_100))  # Compute mean of top 100 for this layer
        return np.mean(layer_means) if layer_means else float('nan')  # Return mean or NaN if empty

    def format_channel_key(channel_key):
        parts = channel_key.split('_')
        if len(parts) == 2:  # Expecting format like "sub18_M05"
            sub_id = parts[0][3:]  # Extract the number after "sub"
            channel = parts[1]
            return f"sub{sub_id}_{channel}"
        else:
            raise ValueError(f"Unexpected channel key format: {channel_key}")

    # For each region, calculate the distribution and the average Spearman correlation
    def compute_region_similarity(layer_top_channels):
        """
        Compute the average distribution of top 100 channels across brain regions, layer by layer.
        """
        if not layer_top_channels or len(layer_top_channels) == 0:
            return {}, []

        region_distributions = []  # Store per-layer distributions

        for top_channels in layer_top_channels:  # Iterate over each layer
            # Ensure top 100 constraint for safety
            top_channels = top_channels[:100]  # Ensure you are working with only the top 100 channels

            region_count = {}  # Collect correlation values per region for this layer

            # Iterate through the top 100 channels and collect correlations by region
            for _, correlation, region_label in top_channels:
                if region_label not in region_count:
                    region_count[region_label] = []
                region_count[region_label].append(correlation)  # Append correlation to the region

            # Compute the mean correlation for each region in this layer
            layer_distribution = {region: np.mean(corrs) for region, corrs in region_count.items()}
            region_distributions.append(layer_distribution)

        # Compute the average distribution across all layers
        avg_region_distribution = {}
        all_regions = set().union(*[dist.keys() for dist in region_distributions])  # All regions across layers
        for region_ in all_regions:
            # Collect layer means, ignoring layers where the region is not present
            values = [dist[region_] for dist in region_distributions if region_ in dist]
            if values:
                avg_region_distribution[region_] = np.mean(values)
            else:
                avg_region_distribution[region_] = 0.0  # or handle as needed

        return region_distributions, avg_region_distribution

    # Write the region similarity to a CSV file
    def write_region_similarity_to_csv(region_similarity, model_name, corpus_name, hemisphere, neuron_type, output_dir):
        # Define the output file path
        output_file = os.path.join(output_dir, f'{model_name}_{corpus_name}_{hemisphere}_{neuron_type}_similarity.csv')

        # Define the region order as per the requirement
        region_order = [
            'A1_L', 'STG_L', 'MTG_L', 'ITG_L', 'Insula_L', 'TPJ_L', 'Temporal_Pole_L', 'Sensorimotor_L',
            'IFG_L', 'MFG_L', 'Hippocampus_L', 'Amygdala_L',
            'A1_R', 'STG_R', 'MTG_R', 'ITG_R', 'Insula_R', 'TPJ_R', 'Temporal_Pole_R', 'Sensorimotor_R',
            'IFG_R', 'MFG_R', 'Hippocampus_R', 'Amygdala_R'
        ]

        # Initialize the similarity list with NaN for all regions
        similarity_list = [np.nan] * len(region_order)

        # Fill the similarity list with actual values from region_similarity if available
        for region_label, similarity_value in region_similarity.items():
            if region_label in region_order:
                index = region_order.index(region_label)
                similarity_list[index] = similarity_value

        # Prepare the DataFrame
        similarity_data = {
            'region_label': region_order,
            'model_region_similarity': similarity_list
        }

        similarity_df = pd.DataFrame(similarity_data)

        # Save to CSV
        similarity_df.to_csv(output_file, index=False)

    # Put it all together for each condition and hemisphere
    if random_effect:
        log_file = 'Results/correlation/search_light_random/output.txt'
    else:
        log_file = 'Results/correlation/search_light/output.txt'
    logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s')

    for condition in conditions:
        block_names = ['Sentence_block1', 'Sentence_block2', 'Phrase_block1', 'Phrase_block2', 'Rand_block1',
                       'Rand_block2']

        for hemisphere in ['L', 'R']:
            significant_df = pd.read_csv(f'SEEG_itpc/{condition_map[corpus_name]}_significant_channel_{hemisphere}.csv')
            if condition == 'sentence':
                # Process significant si, pi, and shared neurons for sentence
                si_neurons = extract_significant_neurons(corpus_name, 'si', random_effect)
                pi_neurons = extract_significant_neurons(corpus_name, 'pi', random_effect)
                shared_neurons = extract_significant_neurons(corpus_name, 'shared', random_effect)

                # Compute RDMs for neurons
                si_rdm = compute_rdm_for_neurons(si_neurons, corpus_name, block_names)
                pi_rdm = compute_rdm_for_neurons(pi_neurons, corpus_name, block_names)
                shared_rdm = compute_rdm_for_neurons(shared_neurons, corpus_name, block_names)

                # Average RDMs for each layer
                avg_si_rdm = average_rdm(si_rdm)
                avg_pi_rdm = average_rdm(pi_rdm)
                avg_shared_rdm = average_rdm(shared_rdm)

                # Correlate with SEEG data
                sub_ids = [sub_id for sub_id in range(1, 29) if sub_id != 15 and sub_id != 21]  # Exclude sub 15 and 21
                seeg_rdm_list = {}
                for sub_id in tqdm(sub_ids, desc=f'Processing subjects for {hemisphere}'):
                    subject_seeg_rdms = compute_rdm_for_seeg(sub_id,
                                                             hemisphere)  # Get RDM for all channels of a subject
                    for channel_key, rdm in subject_seeg_rdms.items():
                        seeg_rdm_list[channel_key] = rdm

                # Correlate model layers with SEEG RDMs
                # Store the results for every layer
                all_layers_si_correlations = []
                all_layers_pi_correlations = []
                all_layers_shared_correlations = []
                all_layers_si_top_channels = []
                all_layers_pi_top_channels = []
                all_layers_shared_top_channels = []

                for layer_idx in tqdm(range(len(avg_si_rdm)), desc='Correlation of model layer RDM with channel RMD:'):
                    itpc_file_path = f'Results/correlation/permutations/itpc_{hemisphere}.hdf5'

                    si_correlations, si_top_channels = correlate_rdms(avg_si_rdm[layer_idx], seeg_rdm_list,
                                                                      itpc_file_path)
                    pi_correlations, pi_top_channels = correlate_rdms(avg_pi_rdm[layer_idx], seeg_rdm_list,
                                                                      itpc_file_path)
                    shared_correlations, shared_top_channels = correlate_rdms(avg_shared_rdm[layer_idx], seeg_rdm_list,
                                                                              itpc_file_path)

                    all_layers_si_correlations.append(si_correlations)
                    all_layers_pi_correlations.append(pi_correlations)
                    all_layers_shared_correlations.append(shared_correlations)
                    all_layers_si_top_channels.append(si_top_channels)
                    all_layers_pi_top_channels.append(pi_top_channels)
                    all_layers_shared_top_channels.append(shared_top_channels)

                    save_correlation_results(si_correlations, si_top_channels, hemisphere, model_name, corpus_name,
                                             output_dir, layer_idx, itpc_file_path, 'si', random_effect)
                    save_correlation_results(pi_correlations, pi_top_channels, hemisphere, model_name, corpus_name,
                                             output_dir, layer_idx, itpc_file_path, 'pi', random_effect)
                    save_correlation_results(shared_correlations, shared_top_channels, hemisphere, model_name,
                                             corpus_name,
                                             output_dir, layer_idx, itpc_file_path, 'shared', random_effect)

                # Step 6: Perform Chi-square test for SI/PI/Shared neurons
                # Initialize dictionaries to hold the final search light ratios for each region
                final_si_search_light = {}
                final_pi_search_light = {}
                final_shared_search_light = {}

                # Loop over each region label and calculate the SI/PI/Shared search light ratio
                for region in significant_df['Region_label'].unique():

                    # Step 6.1: SI search light ratio
                    si_search_light_per_layer = []
                    for layer_idx, layer_si_rdm in enumerate(avg_si_rdm):
                        if layer_si_rdm is None or np.isnan(layer_si_rdm).all():
                            all_layers_si_correlations.append([])
                            all_layers_si_top_channels.append([])
                            continue

                        si_overlap = [
                            item for item in all_layers_si_top_channels[layer_idx]
                            if item[2] == region and
                               format_channel_key(item[0]) in significant_df.loc[
                                   significant_df['Region_label'] == region, 'Exclusive SI channel'
                               ].str.split(', ').explode().values
                        ]
                        si_search_light_per_layer.append(len(si_overlap) / 100)

                    final_si_search_light[region] = np.mean(
                        si_search_light_per_layer) if si_search_light_per_layer else 0

                    # Step 6.2: PI search light ratio
                    pi_search_light_per_layer = []
                    for layer_idx, layer_pi_rdm in enumerate(avg_pi_rdm):
                        if layer_pi_rdm is None or np.isnan(layer_pi_rdm).all():
                            all_layers_pi_correlations.append([])
                            all_layers_pi_top_channels.append([])
                            continue

                        pi_overlap = [
                            item for item in all_layers_pi_top_channels[layer_idx]
                            if item[2] == region and
                               format_channel_key(item[0]) in significant_df.loc[
                                   significant_df['Region_label'] == region, 'Exclusive PI channel'
                               ].str.split(', ').explode().values
                        ]
                        pi_search_light_per_layer.append(len(pi_overlap) / 100)

                    final_pi_search_light[region] = np.mean(
                        pi_search_light_per_layer) if pi_search_light_per_layer else 0

                    # Step 6.3: Shared search light ratio
                    shared_search_light_per_layer = []
                    for layer_idx, layer_shared_rdm in enumerate(avg_shared_rdm):
                        if layer_shared_rdm is None or np.isnan(layer_shared_rdm).all():
                            all_layers_shared_correlations.append([])
                            all_layers_shared_top_channels.append([])
                            continue

                        shared_overlap = [
                            item for item in all_layers_shared_top_channels[layer_idx]
                            if item[2] == region and
                               format_channel_key(item[0]) in significant_df.loc[
                                   significant_df['Region_label'] == region, 'Shared channel'
                               ].str.split(', ').explode().values
                        ]
                        shared_search_light_per_layer.append(len(shared_overlap) / 100)

                    final_shared_search_light[region] = np.mean(
                        shared_search_light_per_layer) if shared_search_light_per_layer else 0

                # ----- Build lists for overall chi-square test using all regions -----

                regions_list = significant_df['Region_label'].unique()

                # 1) For SI:
                obs_counts_si = []
                exp_counts_si = []
                for region in regions_list:
                    obs_counts_si.append(final_si_search_light[region] * 100)
                    if len(significant_df.loc[
                               significant_df['Region_label'] == region, 'Exclusive SI ratio'].values) > 0:
                        exclusive_si_ratio = significant_df.loc[
                            significant_df['Region_label'] == region, 'Exclusive SI ratio'
                        ].values[0]
                    else:
                        exclusive_si_ratio = 0.0
                    exp_counts_si.append(exclusive_si_ratio * 100)

                # Convert to numpy arrays and add epsilon to expected counts
                obs_counts_si = np.array(obs_counts_si, dtype=float)
                exp_counts_si = np.array(exp_counts_si, dtype=float) + epsilon

                sum_obs_si = obs_counts_si.sum()
                sum_exp_si = exp_counts_si.sum()

                # Scale the expected counts to have the same total as observed
                scaled_exp_si = exp_counts_si / sum_exp_si * sum_obs_si
                stat_si, p_value_si = chisquare(f_obs=obs_counts_si, f_exp=scaled_exp_si)

                # 2) For PI:
                obs_counts_pi = []
                exp_counts_pi = []
                for region in regions_list:
                    obs_counts_pi.append(final_pi_search_light[region] * 100)
                    if len(significant_df.loc[
                               significant_df['Region_label'] == region, 'Exclusive PI ratio'].values) > 0:
                        exclusive_pi_ratio = significant_df.loc[
                            significant_df['Region_label'] == region, 'Exclusive PI ratio'
                        ].values[0]
                    else:
                        exclusive_pi_ratio = 0.0
                    exp_counts_pi.append(exclusive_pi_ratio * 100)

                obs_counts_pi = np.array(obs_counts_pi, dtype=float)
                exp_counts_pi = np.array(exp_counts_pi, dtype=float) + epsilon

                sum_obs_pi = obs_counts_pi.sum()
                sum_exp_pi = exp_counts_pi.sum()

                scaled_exp_pi = exp_counts_pi / sum_exp_pi * sum_obs_pi
                stat_pi, p_value_pi = chisquare(f_obs=obs_counts_pi, f_exp=scaled_exp_pi)

                # 3) For Shared:
                obs_counts_shared = []
                exp_counts_shared = []
                for region in regions_list:
                    obs_counts_shared.append(final_shared_search_light[region] * 100)
                    if len(significant_df.loc[significant_df['Region_label'] == region, 'Shared ratio'].values) > 0:
                        exclusive_shared_ratio = significant_df.loc[
                            significant_df['Region_label'] == region, 'Shared ratio'
                        ].values[0]
                    else:
                        exclusive_shared_ratio = 0.0
                    exp_counts_shared.append(exclusive_shared_ratio * 100)

                obs_counts_shared = np.array(obs_counts_shared, dtype=float)
                exp_counts_shared = np.array(exp_counts_shared, dtype=float) + epsilon

                sum_obs_shared = obs_counts_shared.sum()
                sum_exp_shared = exp_counts_shared.sum()

                scaled_exp_shared = exp_counts_shared / sum_exp_shared * sum_obs_shared
                stat_shared, p_value_shared = chisquare(f_obs=obs_counts_shared, f_exp=scaled_exp_shared)

                # ----- Output overall chi-square test results for Sentence condition -----
                if p_value_si < 0.05:
                    print(f"{model_name} is correlated with SEEG in {hemisphere} for SI (overall)!")
                    logging.info(f"{model_name} is correlated with SEEG in {hemisphere} for SI (overall)!")
                else:
                    print(f"{model_name} does not pass significant test in {hemisphere} for SI (overall)!")
                    logging.info(f"{model_name} does not pass significant test in {hemisphere} for SI (overall)!")

                if p_value_pi < 0.05:
                    print(f"{model_name} is correlated with SEEG in {hemisphere} for PI (overall)!")
                    logging.info(f"{model_name} is correlated with SEEG in {hemisphere} for PI (overall)!")
                else:
                    print(f"{model_name} does not pass significant test in {hemisphere} for PI (overall)!")
                    logging.info(f"{model_name} does not pass significant test in {hemisphere} for PI (overall)!")

                if p_value_shared < 0.05:
                    print(f"{model_name} is correlated with SEEG in {hemisphere} for Shared (overall)!")
                    logging.info(f"{model_name} is correlated with SEEG in {hemisphere} for Shared (overall)!")
                else:
                    print(f"{model_name} does not pass significant test in {hemisphere} for Shared (overall)!")
                    logging.info(f"{model_name} does not pass significant test in {hemisphere} for Shared (overall)!")

                # ----- Compute average top 100 Spearman correlations (only if overall significance is met) -----
                if p_value_si < 0.05:
                    top_100_spearman_si = compute_top_100_spearman(all_layers_si_correlations)  # Using top 100 channels
                    region_distribution_si, region_similarity_si = compute_region_similarity(all_layers_si_top_channels)
                    if random_effect:
                        similarity_dir = os.path.join(output_dir, "search_light_random")
                    else:
                        similarity_dir = os.path.join(output_dir, "search_light")

                    write_region_similarity_to_csv(region_similarity_si, model_name, corpus_name, hemisphere, 'SI',
                                                   similarity_dir)
                    print(
                        f"Average top 100 Spearman correlations for {model_name} {condition} SI in {hemisphere}: {top_100_spearman_si}")
                    logging.info(
                        f"Average top 100 Spearman correlations for {model_name} {condition} SI in {hemisphere}: {top_100_spearman_si}")

                if p_value_pi < 0.05:
                    top_100_spearman_pi = compute_top_100_spearman(all_layers_pi_correlations)  # Using top 100 channels
                    region_distribution_pi, region_similarity_pi = compute_region_similarity(all_layers_pi_top_channels)
                    if random_effect:
                        similarity_dir = os.path.join(output_dir, "search_light_random")
                    else:
                        similarity_dir = os.path.join(output_dir, "search_light")
                    write_region_similarity_to_csv(region_similarity_pi, model_name, corpus_name, hemisphere, 'PI',
                                                   similarity_dir)
                    print(
                        f"Average top 100 Spearman correlations for {model_name} {condition} PI in {hemisphere}: {top_100_spearman_pi}")
                    logging.info(
                        f"Average top 100 Spearman correlations for {model_name} {condition} PI in {hemisphere}: {top_100_spearman_pi}")

                if p_value_shared < 0.05:
                    top_100_spearman_shared = compute_top_100_spearman(
                        all_layers_shared_correlations)  # Using top 100 channels
                    region_distribution_shared, region_similarity_shared = compute_region_similarity(
                        all_layers_shared_top_channels)
                    if random_effect:
                        similarity_dir = os.path.join(output_dir, "search_light_random")
                    else:
                        similarity_dir = os.path.join(output_dir, "search_light")
                    write_region_similarity_to_csv(region_similarity_shared, model_name, corpus_name, hemisphere,
                                                   'Shared',
                                                   similarity_dir)
                    print(
                        f"Average top 100 Spearman correlations for {model_name} {condition} Shared in {hemisphere}: {top_100_spearman_shared}")
                    logging.info(
                        f"Average top 100 Spearman correlations for {model_name} {condition} Shared in {hemisphere}: {top_100_spearman_shared}")

            # For Phrase condition, only compute for PI neurons
            if condition == 'phrase':
                # Step 1: Extract significant PI neurons
                pi_neurons = extract_significant_neurons(corpus_name, 'pi', random_effect)
                regions_list = significant_df['Region_label'].unique()

                # Step 2: Compute RDM for PI neurons
                pi_rdm = compute_rdm_for_neurons(pi_neurons, corpus_name, block_names)
                avg_pi_rdm = average_rdm(pi_rdm)

                # Step 3: Correlate PI RDMs with SEEG data
                seeg_rdm_list = {}
                sub_ids = [sub_id for sub_id in range(1, 29) if sub_id != 15 and sub_id != 21]  # Exclude sub 15 and 21
                for sub_id in tqdm(sub_ids, desc='Processing subjects'):
                    seeg_rdms = compute_rdm_for_seeg(sub_id, hemisphere)
                    for channel_key, rdm in seeg_rdms.items():
                        seeg_rdm_list[channel_key] = rdm

                # Step 4: Correlate model layers with SEEG RDMs for PI neurons
                all_layers_pi_correlations = []
                all_layers_pi_top_channels = []

                for layer_idx in range(len(avg_pi_rdm)):
                    if avg_pi_rdm[layer_idx] is None or np.isnan(avg_pi_rdm[layer_idx]).all():
                        all_layers_pi_correlations.append([])
                        all_layers_pi_top_channels.append([])
                        continue
                    itpc_file_path = f'Results/correlation/permutations/itpc_{hemisphere}.hdf5'
                    pi_correlations, pi_top_channels = correlate_rdms(avg_pi_rdm[layer_idx], seeg_rdm_list,
                                                                      itpc_file_path)
                    all_layers_pi_correlations.append(pi_correlations)
                    all_layers_pi_top_channels.append(pi_top_channels)
                    save_correlation_results(pi_correlations, pi_top_channels, hemisphere, model_name, corpus_name,
                                             output_dir, layer_idx, itpc_file_path, 'phrase_pi', random_effect)

                # Step 6: For phrase condition, compute overall chi-square test for PI neurons
                final_pi_search_light = {}
                for region in significant_df['Region_label'].unique():
                    pi_search_light_per_layer = []
                    for layer_idx, layer_pi_rdm in enumerate(avg_pi_rdm):
                        if layer_pi_rdm is None or np.isnan(layer_pi_rdm).all():
                            continue

                        pi_overlap = [
                            item for item in all_layers_pi_top_channels[layer_idx]
                            if item[2] == region and
                               format_channel_key(item[0]) in significant_df.loc[
                                   significant_df['Region_label'] == region, 'Exclusive PI channel'
                               ].str.split(', ').explode().values
                        ]
                        pi_search_light_per_layer.append(len(pi_overlap) / 100)
                    final_pi_search_light[region] = np.mean(
                        pi_search_light_per_layer) if pi_search_light_per_layer else 0

                # Build lists for overall chi-square test for phrase PI
                obs_counts_pi_phrase = []
                exp_counts_pi_phrase = []
                for region in regions_list:
                    obs_counts_pi_phrase.append(final_pi_search_light[region] * 100)
                    if len(significant_df.loc[
                               significant_df['Region_label'] == region, 'Exclusive PI ratio'].values) > 0:
                        exclusive_pi_ratio = significant_df.loc[
                            significant_df['Region_label'] == region, 'Exclusive PI ratio'
                        ].values[0]
                    else:
                        exclusive_pi_ratio = 0.0
                    exp_counts_pi_phrase.append(exclusive_pi_ratio * 100)

                obs_counts_pi_phrase = np.array(obs_counts_pi_phrase, dtype=float)
                exp_counts_pi_phrase = np.array(exp_counts_pi_phrase, dtype=float) + epsilon

                sum_obs_pi_phrase = obs_counts_pi_phrase.sum()
                sum_exp_pi_phrase = exp_counts_pi_phrase.sum()

                scaled_exp_pi_phrase = exp_counts_pi_phrase / sum_exp_pi_phrase * sum_obs_pi_phrase
                stat_pi_phrase, p_value_pi_phrase = chisquare(f_obs=obs_counts_pi_phrase, f_exp=scaled_exp_pi_phrase)

                if p_value_pi_phrase < 0.05:
                    print(f"{model_name} is correlated with SEEG in {hemisphere} for PI (phrase condition)!")
                    logging.info(f"{model_name} is correlated with SEEG in {hemisphere} for PI (phrase condition)!")
                    top_100_spearman_pi = compute_top_100_spearman(all_layers_pi_correlations)  # Using top 100 channels
                    region_distribution_pi, region_similarity_pi = compute_region_similarity(all_layers_pi_top_channels)
                    if random_effect:
                        similarity_dir = os.path.join(output_dir, "search_light_random")
                    else:
                        similarity_dir = os.path.join(output_dir, "search_light")
                    write_region_similarity_to_csv(region_similarity_pi, model_name, corpus_name, hemisphere, 'PI',
                                                   similarity_dir)
                    print(
                        f"Average top 100 Spearman correlations for {model_name} {condition} PI in {hemisphere}: {top_100_spearman_pi}")
                    logging.info(
                        f"Average top 100 Spearman correlations for {model_name} {condition} PI in {hemisphere}: {top_100_spearman_pi}")
                else:
                    print(f"{model_name} does not pass significant test in {hemisphere} for PI (phrase condition)!")
                    logging.info(
                        f"{model_name} does not pass significant test in {hemisphere} for PI (phrase condition)!")


def main():
    """Main function to orchestrate the entire analysis pipeline."""
    # Part 1: LLM activation extraction
    models = {
        'gpt2-large-chinese': gpt2_activation,
        'Llama-2-7b-hf': llama2_activation,
        'Llama-3.1-8B': llama3_1_activation,
        'gemma-2b': gemma_activation,
        'gemma-2-9b': gemma_activation,
        'GLM-4-9b': glm_activation,
    }
    corpus_paths = ['data/SEEG_Sentence.csv', 'data/SEEG_Phrase.csv', 'data/SEEG_Rand.csv']
    condition_map = {
        'SEEG_Sentence': 'sentence',
        'SEEG_Phrase': 'phrase',
        'SEEG_Rand': 'rand'
    }
    significant_channel_paths = {
        'phrase': 'SEEG_itpc/phrase_significant_channel.csv',
        'sentence': 'SEEG_itpc/sentence_significant_channel.csv'
    }
    seeg_data_dir = 'data_SEEG'
    activation_data_dir = 'Results/correlation/activations'
    output_dir = 'Results/correlation'

    # Activation extraction for each model
    for model_name, activation_function in models.items():
        print(f"Processing model: {model_name}")
        for corpus_path in corpus_paths:
            activation_function(corpus_path, model_name=model_name)

    # Permutation test for each model (now block-specific)
    for model_name in models.keys():
        print(f"Running permutation test for model: {model_name}")
        for corpus_path in corpus_paths[:-1]:  # Exclude Rand condition
            corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
            # Run permutation tests for both block1 and block2
            permutation_test(model_name, corpus_name, input_dir=activation_data_dir, output_dir=output_dir)

    # Compute significant neurons using z-scores (now block-specific)
    confidence_interval = 95  # Modify this value as needed
    for model_name in models.keys():
        for corpus_path in corpus_paths[:-1]:  # Exclude Rand condition
            corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
            print(f"Calculating significant neurons for model: {model_name}, corpus: {corpus_name}")
            significant_neurons_zscore(model_name, corpus_name, confidence_interval, activation_data_dir,
                                           output_dir)

    # Part 2: sEEG Data Processing
    # Define the subject IDs using range and exclude specific IDs if necessary
    sub_ids = [sub_id for sub_id in range(1, 29) if sub_id != 15 and sub_id != 21]  # Exclude sub 15 amd 21

    base_dir = 'data_SEEG'  # Base directory for SEEG data
    region_file = 'data_SEEG/Region.csv'  # Region information file

    # Load the region data from the CSV
    region_df = pd.read_csv(region_file)

    # Directory for storing the permutation results
    output_dir = 'Results/correlation/permutations'

    # Initialize results to store ITPC results for left and right hemispheres
    all_results = {}

    # Process SEEG data for each subject
    for sub_id in tqdm(sub_ids, desc='Processing subjects:'):
        sub_folder = os.path.join(base_dir, f'Sub{sub_id}', 'matdata_EEG_epoch')

        # Process SEEG data for each subject, calculate ITPC, and run permutation tests
        subject_results = process_seeg(sub_folder, sub_id, region_df, output_dir)

        all_results[sub_id] = subject_results

    # Calculate significant channels for all subjects
    print("ITPC results for two blocks saved")

    # # Part 3: Correlation Analysis
    for model_name, activation_func in models.items():
        print(f"Processing model: {model_name}")

        # Loop through each corpus
        for corpus_path in corpus_paths:
            corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
            condition = condition_map[corpus_name]

            # Skip rand condition as per your instruction
            if condition == 'rand':
                continue

            print(f"Processing corpus: {corpus_name} for model: {model_name}")

            # Call the calculate_correlation function for each model and condition
            calculate_correlation(
                model_name=model_name,
                corpus_name=corpus_name,
                conditions=[condition],
                activation_data_dir=activation_data_dir,
                output_dir=output_dir,
                random_effect=False  # Set to True to enable random effect experiments
            )


# Run the main function
if __name__ == "__main__":
    main()
