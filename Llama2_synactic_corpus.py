from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ast
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft, rfftfreq
import seaborn as sns
import matplotlib.cm as cm
import h5py
import re
from scipy.signal import get_window, butter, filtfilt
from scipy.stats import ttest_ind
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import gridspec
from definitions import statistic_significance, compare_cross_language_neurons, permutation_significant, significant_neurons_zscore
import itertools


def _split_units_control_c(phrase, is_chinese):
    """Split phrase into character units for Chinese or word units for English."""
    if is_chinese:
        return [char for char in phrase if char.strip()]

    cleaned = re.sub(r'[^\w\s]', '', phrase)
    return [token for token in cleaned.split() if token]


def _join_units_control_c(units, is_chinese):
    """Join units back together - no spaces for Chinese, spaces for English."""
    return ''.join(units) if is_chinese else ' '.join(units)


def _shuffle_units_random(units, rng, max_attempts=200):
    """Randomly shuffle units while avoiding preserving original adjacent pairs."""
    if len(units) <= 1:
        return list(units)

    original_indices = list(range(len(units)))
    forbidden_pairs = {(i, i + 1) for i in range(len(units) - 1)}

    # Try randomized permutations first
    for _ in range(max_attempts):
        candidate = original_indices[:]
        rng.shuffle(candidate)
        if candidate == original_indices:
            continue
        if all((candidate[i], candidate[i + 1]) not in forbidden_pairs for i in range(len(candidate) - 1)):
            return [units[idx] for idx in candidate]

    # Collect all valid permutations (phrases are short, so this is feasible)
    valid_orders = [
        perm for perm in itertools.permutations(original_indices)
        if list(perm) != original_indices
        and all((perm[i], perm[i + 1]) not in forbidden_pairs for i in range(len(perm) - 1))
    ]

    if valid_orders:
        chosen = rng.choice(valid_orders)
        return [units[idx] for idx in chosen]

    # As a last resort, reverse the order if it breaks any adjacency
    reversed_indices = list(reversed(original_indices))
    if reversed_indices != original_indices and any(
        (reversed_indices[i], reversed_indices[i + 1]) not in forbidden_pairs
        for i in range(len(reversed_indices) - 1)
    ):
        return [units[idx] for idx in reversed_indices]

    raise ValueError('Unable to find a permutation that breaks original adjacencies.')



def shuffle_and_recombine_control(words, is_chinese, random_seed=None):
    """Shuffle units within each phrase while maintaining language-specific structure."""
    rng = random.Random(random_seed)
    shuffled_phrases = []

    for phrase in words:
        units = _split_units_control_c(phrase, is_chinese)
        if not units:
            shuffled_phrases.append(phrase)
            continue

        new_units = _shuffle_units_random(units, rng)
        shuffled_phrases.append(_join_units_control_c(new_units, is_chinese))

    return shuffled_phrases


def shuffle_and_recombine_control_experiment(words):
    """Shuffle entire phrases randomly (experimental condition)."""
    shuffled_phrases = words.copy()
    random.shuffle(shuffled_phrases)
    return shuffled_phrases


def handle_file(file_path, output_dir, split_type, strategy):
    """Process a single corpus file by shuffling and computing activations."""
    is_chinese = "Chinese" in file_path

    df = pd.read_csv(file_path, header=None)
    column_index = 0 if len(df.columns) == 1 else 1
    words = df[column_index].dropna().astype(str).tolist()

    n_shuffles = 0
    shuffle_results = []

    activations_accumulator = {}
    grouped_means = {}

    for shuffle_index in tqdm(range(num_shuffles), desc=f"Shuffling and processing ({strategy})"):
        if strategy == "experiment":
            shuffled_words = shuffle_and_recombine_control_experiment(words)
        else:  # control-B
            shuffled_words = shuffle_and_recombine_control(words, is_chinese)

        shuffle_results.append(shuffled_words)
        activations = process_text_and_accumulate_activations(shuffled_words, tokenizer, model, is_chinese)

        for layer_index, layer_activations in activations.items():
            if layer_index not in activations_accumulator:
                activations_accumulator[layer_index] = []

            activations_accumulator[layer_index].append(layer_activations)

        n_shuffles += 1

        # Save grouped means every 10 shuffles to manage memory
        if n_shuffles % 10 == 0:
            keys = list(activations_accumulator.keys())
            for layer_index in keys:
                group_mean = np.mean(activations_accumulator[layer_index], axis=0)
                layer_name = f'Layer{layer_index + 1}'
                if layer_name not in grouped_means:
                    grouped_means[layer_name] = []
                grouped_means[layer_name].append(group_mean)

            activations_accumulator = {layer_index: [] for layer_index in keys}

    # Save activations to HDF5 file
    hdf5_path = os.path.join(output_dir, 'activations', strategy, f"{split_type}_activations.hdf5")

    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    with h5py.File(hdf5_path, 'w') as hdf:
        for layer_name in grouped_means:
            layer_group = hdf.create_group(layer_name)
            for i, group_mean in enumerate(grouped_means[layer_name]):
                layer_group.create_dataset(f'Neuron_group_{i + 1}', data=group_mean)

    # Save shuffle results to CSV
    df_shuffle = pd.DataFrame(shuffle_results).transpose()
    df_shuffle.columns = [f'Shuffle_{i + 1}' for i in range(num_shuffles)]
    shuffle_output_path = os.path.join(output_dir, strategy, os.path.basename(file_path).replace(".csv", "_shuffle.csv"))
    df_shuffle.to_csv(shuffle_output_path, index=False, encoding='utf_8_sig')



def _normalize_units(words, is_chinese):
    """Normalize text and record unit spans for Chinese characters or English words."""

    combined_builder = []
    unit_spans = []
    cursor = 0

    for phrase in words:
        cleaned = re.sub(r'[^\w\s]', '', phrase)

        if is_chinese:
            units = [ch for ch in cleaned if ch.strip()]
            for unit in units:
                start = cursor
                combined_builder.append(unit)
                cursor += len(unit)
                unit_spans.append((start, cursor))
        else:
            units = [tok for tok in cleaned.split() if tok]
            for unit in units:
                if combined_builder:
                    combined_builder.append(' ')
                    cursor += 1
                start = cursor
                combined_builder.append(unit)
                cursor += len(unit)
                unit_spans.append((start, cursor))

    combined_text = ''.join(combined_builder)
    return combined_text, unit_spans


def process_text_and_accumulate_activations(words, tokenizer, model, is_chinese):
    """Extract neural activations from model for the given text."""
    combined_text, unit_spans = _normalize_units(words, is_chinese)

    if not combined_text:
        raise ValueError("Input text is empty after preprocessing; cannot compute activations.")

    encoding = tokenizer(
        combined_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    if hasattr(encoding, "encodings"):
        offset_mapping = encoding.encodings[0].offsets
    else:
        offset_mapping = encoding["offset_mapping"]

    input_tensor = torch.tensor([input_ids]).to(device)

    mlp_activations = []

    # Define the hook function to capture intermediate layer activations
    def hook_fn(module, input, output):
        # Capturing the combined activations of gate_proj and up_proj
        gate_output = module.gate_proj(input[0])
        up_output = module.up_proj(input[0])
        activation_output = module.act_fn(gate_output) * up_output
        mlp_activations.append(activation_output.to(torch.float32).detach().cpu().numpy())

    # Register hooks to the MLP layer in each transformer block
    hooks = []
    for block in model.model.layers:
        hook = block.mlp.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        model(input_tensor, output_hidden_states=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not mlp_activations:
        raise ValueError("No MLP activations are captured.")

    # Aggregate activations by unit spans
    activations_dict = {}
    for layer_index, layer_activations in enumerate(mlp_activations):
        activations_layer = layer_activations.squeeze(0)
        aggregated = np.zeros((len(unit_spans), activations_layer.shape[1]))

        for unit_idx, (unit_start, unit_end) in enumerate(unit_spans):
            token_indices = [
                i
                for i, (tok_start, tok_end) in enumerate(offset_mapping)
                if tok_end > tok_start and not (tok_end <= unit_start or tok_start >= unit_end)
            ]

            if not token_indices:
                raise ValueError(
                    f"No tokens found for unit span ({unit_start}, {unit_end}). Check preprocessing assumptions."
                )

            unit_vectors = activations_layer[token_indices]
            aggregated[unit_idx] = unit_vectors.mean(axis=0)

        activations_dict[layer_index] = aggregated

    return activations_dict


def bonferroni_correction(p_vals, alpha=0.05):
    """Apply Bonferroni correction to p-values and return thresholds."""
    n = len(p_vals)
    corrected_pvals = np.minimum(np.array(p_vals) * n, 1.0)  # Multiplying p-values by the number of tests
    return corrected_pvals


def clean_neuron_list(neuron_list):
    """Remove extraneous characters and convert to set of integers."""
    cleaned_list = re.sub(r'[^\d,]', '', neuron_list)  # Remove all characters except digits and commas
    return set(map(int, cleaned_list.split(','))) if cleaned_list else set()



def compare_language_specific_neurons(output_dir, split_type):
    """Compare overlap between significant neurons and language-specific neurons."""
    # Determine the correct neuron file based on the split_type
    if split_type == 'ssvo':
        language_neurons_file = os.path.join(output_dir, 'heatmap', 'english_neurons.csv')
    else:
        language_neurons_file = os.path.join(output_dir, 'heatmap', 'chinese_neurons.csv')

    # Load the significant_neurons count and language-specific neurons data
    significant_count_file = os.path.join(output_dir, 'heatmap', f'{split_type}_significant_count.csv')
    significant_count_df = pd.read_csv(significant_count_file)
    language_neurons_df = pd.read_csv(language_neurons_file)

    # Merge the two datasets based on the Layer column
    merged_df = pd.merge(significant_count_df, language_neurons_df, on='Layer', suffixes=('_sig', '_language'))

    # Function to clean and convert neuron lists from strings to sets of integers
    def clean_neuron_list(neuron_list):
        if isinstance(neuron_list, str):
            neuron_list = neuron_list.strip('[]').split(',')
            return set(int(neuron.strip().strip("'")) for neuron in neuron_list if neuron.strip())
        elif isinstance(neuron_list, list):
            return set(neuron_list)
        else:
            return set()

    # Clean the neuron lists
    merged_df['significant_si_neurons'] = merged_df['significant_si_neurons'].apply(clean_neuron_list)
    merged_df['significant_pi_neurons'] = merged_df['significant_pi_neurons'].apply(clean_neuron_list)
    merged_df['Neuron Indices'] = merged_df['Neuron Indices'].apply(clean_neuron_list)

    # Calculate overlap between significant and language-specific neurons
    merged_df['si_language_overlap'] = merged_df.apply(lambda row: len(row['significant_si_neurons'] & row['Neuron Indices']), axis=1)
    merged_df['pi_language_overlap'] = merged_df.apply(lambda row: len(row['significant_pi_neurons'] & row['Neuron Indices']), axis=1)

    # Visualization: Overlap of SI and PI Neurons with Language-specific Neurons across Layers
    plt.figure(figsize=(14, 8))

    # SI overlap visualization
    plt.subplot(2, 1, 1)
    sns.barplot(x='Layer', y='si_language_overlap', data=merged_df, hue='Layer', palette='Blues_d', legend=False)
    plt.legend([], [], frameon=False)
    if split_type == 'ssvo':
        plt.title(f'Overlap of Significant $\\mathit{{si}}$ Neurons with English-specific Neurons Across Layers')
    else:
        plt.title(f'Overlap of Significant $\\mathit{{si}}$ Neurons with Chinese-specific Neurons Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Number of Overlapping Neurons')
    plt.xticks(rotation=90)

    # PI overlap visualization
    plt.subplot(2, 1, 2)
    sns.barplot(x='Layer', y='pi_language_overlap', data=merged_df, hue='Layer', palette='Reds_d', legend=False)
    plt.legend([], [], frameon=False)
    if split_type == 'ssvo':
        plt.title(f'Overlap of Significant $\\mathit{{pi}}$ Neurons with English-specific Neurons Across Layers')
    else:
        plt.title(f'Overlap of Significant $\\mathit{{pi}}$ Neurons with Chinese-specific Neurons Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Number of Overlapping Neurons')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap/statistic/{split_type}_language_specific.png'), dpi=300)
    plt.close()


def process_all_files(input_dir, output_dir):
    """Process all corpus files with both experimental and control strategies."""
    strategies = ["experiment", "control-B"]
    for strategy in strategies:
        os.makedirs(os.path.join(output_dir, strategy), exist_ok=True)

    # Find relevant corpus files
    files = [filename for filename in os.listdir(input_dir)
             if filename.endswith(".csv") and any(suffix in filename for suffix in ["Chinese_syntactic_corpus", "English_syntactic_corpus"])]

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        split_type = filename.split('_')[-1].split('.')[0]

        hdf5_paths = {}
        for strategy in strategies:
            print(f"Processing {filename} for {strategy} strategy")
            handle_file(input_path, output_dir, split_type, strategy)
            hdf5_paths[strategy] = os.path.join(output_dir, 'activations', strategy, f"{split_type}_activations.hdf5")

        # Run statistical analysis
        permutation_significant(hdf5_paths, output_dir, split_type)
        significant_neurons_zscore(hdf5_paths, output_dir, split_type)
        statistic_significance(output_dir, split_type)

        # Compare cross-language neurons
        english_file = os.path.join(output_dir, 'heatmap', 'English_syntactic_corpus_significant_count.csv')
        chinese_file = os.path.join(output_dir, 'heatmap', 'Chinese_syntactic_corpus_significant_count.csv')
        compare_cross_language_neurons(english_file, chinese_file, output_dir)




# Configuration parameters
num_shuffles = 100

BASE_MODEL_PATH = 'models/Llama2'
BASE_OUTPUT_PATH = 'Results/Llama2'

# Find all available Llama2 model directories
llama2_models = [model for model in os.listdir(BASE_MODEL_PATH) if os.path.isdir(os.path.join(BASE_MODEL_PATH, model))]

# Device selection for GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda:1")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
input_dir = 'data'

# Process each Llama2 model
for model_name in llama2_models:

    print('Processing model:', model_name)
    if 'Llama-2-7b' in model_name:
        tokenizer_path = os.path.join(BASE_MODEL_PATH, model_name)
        output_dir = os.path.join(BASE_OUTPUT_PATH, model_name, 'fft')

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=True,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype=torch.bfloat16, output_hidden_states=True).to(device)

        # Process all corpus files
        process_all_files(input_dir, output_dir)
