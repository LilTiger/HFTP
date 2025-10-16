import ast
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft, rfftfreq
import seaborn as sns
import h5py
from matplotlib.colors import ListedColormap, BoundaryNorm
import re
from matplotlib.ticker import FuncFormatter
from scipy.signal import hilbert
from statsmodels.stats.multitest import multipletests




def clean_neuron_list(neuron_list):
    # Remove extraneous characters and convert to set of integers
    cleaned_list = re.sub(r'[^\d,]', '', neuron_list)  # Remove all characters except digits and commas
    return set(map(int, cleaned_list.split(','))) if cleaned_list else set()



# Function to create stacked bar chart showing proportions of exclusive si/pi neurons and shared neurons for each layer
def statistic_significance(output_dir, split_type):
    # Define the path to the significant neurons CSV file
    csv_file_path = os.path.join(output_dir, 'heatmap', f'{split_type}_significant_count.csv')

    # Load the CSV file
    significant_count_df = pd.read_csv(csv_file_path)

    # Extract significant neurons for sentence-level (1Hz) and phrase-level (2Hz)
    layers = significant_count_df['Layer']
    significant_si_neurons = significant_count_df['exclusive_si_neurons'].apply(
        lambda x: clean_neuron_list(x) if isinstance(x, str) else set())
    significant_pi_neurons = significant_count_df['exclusive_pi_neurons'].apply(
        lambda x: clean_neuron_list(x) if isinstance(x, str) else set())
    shared_neurons = significant_count_df['shared_neurons'].apply(
        lambda x: clean_neuron_list(x) if isinstance(x, str) else set())

    # Calculate Dice Similarity Matrix
    def compute_dice_matrix(significant_neurons):
        num_layers = len(significant_neurons)
        dice_matrix = np.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(num_layers):
                intersection = len(significant_neurons[i].intersection(significant_neurons[j]))
                size_i = len(significant_neurons[i])
                size_j = len(significant_neurons[j])
                if size_i + size_j == 0:
                    dice_matrix[i, j] = 0  # Avoid division by zero
                else:
                    dice_matrix[i, j] = 2 * intersection / (size_i + size_j)

        return dice_matrix

    hdf5_path = os.path.join(output_dir, 'activations', 'experiment', f"{split_type}_activations.hdf5")
    hdf = h5py.File(hdf5_path, 'r')
    first_layer_name = list(hdf.keys())[0]  # Get the name of the first layer (e.g., 'Layer1')
    num_neurons = hdf[first_layer_name]['Neuron_group_1'].shape[1]  # All layers have the same number of neurons

    # Calculate the number of neurons in each category for each layer
    exclusive_sentence_counts = [len(neurons) for neurons in significant_si_neurons]
    exclusive_phrase_counts = [len(neurons) for neurons in significant_pi_neurons]
    intersection_counts = [len(neurons) for neurons in shared_neurons]

    # Calculate proportions of each category per layer
    exclusive_si_proportions = [exclusive_sentence_counts[i] / num_neurons for i in range(len(layers))]
    exclusive_pi_proportions = [exclusive_phrase_counts[i] / num_neurons for i in range(len(layers))]
    shared_proportions = [intersection_counts[i] / num_neurons for i in range(len(layers))]

    # Plot customized stacked bar chart of proportions
    df = pd.DataFrame({
        'Layer': layers,
        'Exclusive SI Neurons': exclusive_si_proportions,
        'Shared Neurons': shared_proportions,
        'Exclusive PI Neurons': exclusive_pi_proportions
    })
    df = df.set_index('Layer')
    df = df[['Exclusive PI Neurons', 'Shared Neurons', 'Exclusive SI Neurons']]

    colors = ['#1ECBE1', '#1EE196', '#1E6AE1']
    ax = df.plot(kind='bar', stacked=True, color=colors, figsize=(14, 8), width=0.8)

    plt.xlabel('Layer', fontsize=28)
    plt.ylabel('Proportion', fontsize=28)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    # Set x-ticks to show every third layer
    ax.set_xticks(range(0, len(layers), 3))
    ax.set_xticklabels(layers[::3], rotation=60, fontsize=24)
    plt.legend(['Exclusive $phrase$ Neurons', '$sentence & phrase$ Neurons', 'Exclusive $sentence$ Neurons'], fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Set the desired font size
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap/statistic/{split_type}_stacked_bar_chart_proportions.png', dpi=300)
    plt.show()
    # Adjusting the function to ensure that shared neurons' proportions are plotted as positive values

# Parse neuron list function
def parse_neuron_list(neuron_list_str):
    """Convert neuron list from string to a set of integers."""
    if neuron_list_str == '[]':
        return set()
    return set(map(int, ast.literal_eval(neuron_list_str)))


# Compare cross-language neurons function
def compare_cross_language_neurons(english_file, chinese_file, output_dir):
    # Load the CSV files
    ssvoe_data = pd.read_csv(english_file)
    ssvoc_data = pd.read_csv(chinese_file)

    # Parse neuron lists in both datasets
    ssvoe_data['exclusive_si_neurons'] = ssvoe_data['exclusive_si_neurons'].apply(parse_neuron_list)
    ssvoe_data['exclusive_pi_neurons'] = ssvoe_data['exclusive_pi_neurons'].apply(parse_neuron_list)
    ssvoc_data['exclusive_si_neurons'] = ssvoc_data['exclusive_si_neurons'].apply(parse_neuron_list)
    ssvoc_data['exclusive_pi_neurons'] = ssvoc_data['exclusive_pi_neurons'].apply(parse_neuron_list)

    # Layers sorting and initialization
    layers = sorted(ssvoe_data['Layer'].unique())

    # Determine number of neurons per layer for the specific model
    hdf5_path = os.path.join(output_dir, 'activations', 'experiment', f"ssvoe_activations.hdf5")
    hdf = h5py.File(hdf5_path, 'r')

    # Convert the layer number to a string format like 'Layer1', 'Layer2', etc.
    num_neurons_per_layer = [hdf[f'Layer{layer}']['Neuron_group_1'].shape[1] for layer in layers]

    # Initialize lists to store neuron proportions for each layer
    exclusive_en_si_prop = []
    shared_si_prop = []
    exclusive_zh_si_prop = []
    exclusive_en_pi_prop = []
    shared_pi_prop = []
    exclusive_zh_pi_prop = []

    # Iterate over layers
    for layer_num in layers:
        # Get the SI and PI neurons for both languages, checking if the layer exists
        ssvoe_layer_data = ssvoe_data.loc[ssvoe_data['Layer'] == layer_num]
        ssvoc_layer_data = ssvoc_data.loc[ssvoc_data['Layer'] == layer_num]

        if not ssvoe_layer_data.empty and not ssvoc_layer_data.empty:
            # Extract the neuron lists for SI and PI
            ssvoe_si_neurons = ssvoe_layer_data['exclusive_si_neurons'].values[0]
            ssvoc_si_neurons = ssvoc_layer_data['exclusive_si_neurons'].values[0]
            ssvoe_pi_neurons = ssvoe_layer_data['exclusive_pi_neurons'].values[0]
            ssvoc_pi_neurons = ssvoc_layer_data['exclusive_pi_neurons'].values[0]

            # Calculate total neurons for the layer
            total_neurons = num_neurons_per_layer[layer_num - 1]

            # Identify shared and exclusive neurons
            shared_si_neurons = ssvoe_si_neurons & ssvoc_si_neurons
            shared_pi_neurons = ssvoe_pi_neurons & ssvoc_pi_neurons

            # Calculate proportions for exclusive and shared neurons
            exclusive_en_si_prop.append(len(ssvoe_si_neurons - shared_si_neurons) / total_neurons)
            shared_si_prop.append(len(shared_si_neurons) / total_neurons)
            exclusive_zh_si_prop.append(len(ssvoc_si_neurons - shared_si_neurons) / total_neurons)

            exclusive_en_pi_prop.append(len(ssvoe_pi_neurons - shared_pi_neurons) / total_neurons)
            shared_pi_prop.append(len(shared_pi_neurons) / total_neurons)
            exclusive_zh_pi_prop.append(len(ssvoc_pi_neurons - shared_pi_neurons) / total_neurons)
        else:
            # Handle cases where the layer doesn't exist in one of the datasets
            exclusive_en_si_prop.append(0)
            shared_si_prop.append(0)
            exclusive_zh_si_prop.append(0)
            exclusive_en_pi_prop.append(0)
            shared_pi_prop.append(0)
            exclusive_zh_pi_prop.append(0)

    # Plot the proportions for SI and PI neurons
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    bar_width = 0.4  # Increase the bar width
    x = np.arange(len(layers))

    # Offset between bars to prevent overlap
    offset = bar_width / 2   # A small extra offset to prevent overlap

    # Plot for SI neurons
    ax1.bar(x - offset, exclusive_en_si_prop, width=bar_width, label='Exclusive English $sentence$ Neurons',
            color='#00A664', edgecolor='black')
    ax1.bar(x + offset, exclusive_zh_si_prop, width=bar_width, label='Exclusive Chinese $sentence$ Neurons',
            color='#5086C4', edgecolor='black')
    ax1.bar(x, np.abs(shared_si_prop), width=bar_width, color='#DCE125', alpha=0.7,
            label='Chinese&English $sentence$ Neurons', edgecolor='black')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Adding gridlines and a reference line at y=0
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel('Proportion', fontsize=28)
    ax1.set_xticks(x[::3])  # Set x-ticks to show every third layer
    ax1.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 3)], fontsize=24,
                        rotation=45)  # Update x-tick labels
    ax1.tick_params(axis='y', labelsize=24)
    ax1.legend(loc='upper left', fontsize=26)

    # Plot for PI neurons
    ax2.bar(x - offset, exclusive_en_pi_prop, width=bar_width, label='Exclusive English $phrase$ Neurons',
            color='#00A664', edgecolor='black')
    ax2.bar(x + offset, exclusive_zh_pi_prop, width=bar_width, label='Exclusive Chinese $phrase$ Neurons',
            color='#5086C4', edgecolor='black')
    ax2.bar(x, np.abs(shared_pi_prop), width=bar_width, color='#DCE125', alpha=0.7,
            label='Chinese&English $phrase$ Neurons', edgecolor='black')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))


    # Adding gridlines and a reference line at y=0
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel('Layer', fontsize=28)
    ax2.set_ylabel('Proportion', fontsize=28)
    ax2.set_xticks(x[::3])  # Set x-ticks to show every third layer
    ax2.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 3)], fontsize=24,
                        rotation=45)  # Update x-tick labels
    ax2.tick_params(axis='y', labelsize=24)
    ax2.legend(loc='upper left', fontsize=26)

    plt.tight_layout()
    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'heatmap/statistic/cross_language_neurons.png'), dpi=300)
    plt.show()


def permutation_significant(hdf5_paths, output_dir, split_type,
                            sample_interval=0.25):
    groups = ['experiment', 'control-B']
    num_permutations = 1000

    hdf_files = {g: h5py.File(hdf5_paths[g], 'r') for g in groups}

    layers = sorted(hdf_files['experiment'].keys(),
                    key=lambda x: int(x.replace('Layer', '')))
    num_layers   = len(layers)
    num_neurons  = hdf_files['experiment'][layers[0]]['Neuron_group_1'].shape[1]

    si_matrix = np.zeros((num_layers, num_neurons))
    pi_matrix = np.zeros((num_layers, num_neurons))

    significant_si_counts, significant_pi_counts, shared_counts = [], [], []
    significant_si_neurons, significant_pi_neurons, shared_neurons = [], [], []

    if split_type == '8-natural' or split_type == '8-naturale' or split_type == '8-zhwiki' or split_type == '8-enwiki':
        si_freq = 4.0 / 8
        pi_freq_list = [4.0 / k for k in range(2, 8)]
    elif split_type == '9-natural' or split_type == '9-naturale':
        si_freq = 4.0 / 9
        pi_freq_list = [4.0 / k for k in range(2, 9)]
    elif split_type == '8-syllable':
        si_freq = 0.5
        pi_freq_list = [1.0]
    else:
        si_freq = 1.0
        pi_freq_list = [2.0]

    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        layer_name = layers[layer_idx]
        experiment_ffts = {n: [] for n in range(num_neurons)}

        for n in range(num_neurons):
            acts = [hdf_files['experiment'][layer_name][f'Neuron_group_{g+1}'][:, n]
                    for g in range(10)]
            avg_exp = np.mean(acts, axis=0)
            exp_fft = rfft(avg_exp); exp_fft[0] = 0
            experiment_ffts[n].append(np.abs(exp_fft))

            if n == 0:
                freqs = rfftfreq(len(avg_exp), d=sample_interval)

            real_si = np.mean([fft[np.argmin(np.abs(freqs - si_freq))]
                               for fft in experiment_ffts[n]])

            # Modified keep neuron if ANY phrase frequency passes 95-percentile threshold
            real_pi_vals = [np.mean([fft[np.argmin(np.abs(freqs - p))]
                                     for fft in experiment_ffts[n]])
                            for p in pi_freq_list]

            perm_si, perm_pi_dists = [], {p: [] for p in pi_freq_list}
            for _ in range(num_permutations):
                shuf = np.random.permutation(avg_exp)
                s_fft = rfft(shuf); s_fft[0] = 0
                perm_si.append(np.abs(s_fft[np.argmin(np.abs(freqs - si_freq))]))
                for p in pi_freq_list:
                    perm_pi_dists[p].append(np.abs(s_fft[np.argmin(np.abs(freqs - p))]))

            upper_si = np.percentile(perm_si, 95)
            upper_pi = {p: np.percentile(perm_pi_dists[p], 95) for p in pi_freq_list}

            if real_si > upper_si:
                si_matrix[layer_idx, n] = real_si

            # Modified pass if at least ONE phrase frequency exceeds threshold
            if any(real > upper_pi[p]
                   for real, p in zip(real_pi_vals, pi_freq_list)):
                pi_matrix[layer_idx, n] = max(real_pi_vals)

        sig_si = np.where(si_matrix[layer_idx] != 0)[0]
        sig_pi = np.where(pi_matrix[layer_idx] != 0)[0]
        shared = list(set(sig_si) & set(sig_pi))

        significant_si_neurons.append(sig_si)
        significant_pi_neurons.append(sig_pi)
        shared_neurons.append(shared)
        significant_si_counts.append(len(sig_si))
        significant_pi_counts.append(len(sig_pi))
        shared_counts.append(len(shared))

    df = pd.DataFrame({
        'Layer': range(1, num_layers + 1),
        'significant_si_neurons': [list(map(str, sorted(n))) for n in significant_si_neurons],
        'number_of_si_neurons': significant_si_counts,
        'significant_pi_neurons': [list(map(str, sorted(n))) for n in significant_pi_neurons],
        'number_of_pi_neurons': significant_pi_counts,
        'shared_neurons': [list(map(str, sorted(n))) for n in shared_neurons],
        'number_of_shared_neurons': shared_counts
    })
    os.makedirs(os.path.join(output_dir, 'heatmap'), exist_ok=True)
    df.to_csv(f'{output_dir}/heatmap/{split_type}_permutation_significant_count.csv',
              index=False)


def significant_neurons_zscore(hdf5_paths, output_dir, split_type,
                        sample_interval=0.25, bin_size=10, z_threshold=2):

    groups = ['experiment', 'control-B']
    hdf_files = {g: h5py.File(hdf5_paths[g], 'r') for g in groups}

    csv_path = os.path.join(output_dir, 'heatmap',
                            f'{split_type}_permutation_significant_count.csv')
    sig_df = pd.read_csv(csv_path)

    layers = sorted(hdf_files['experiment'].keys(),
                    key=lambda x: int(x.replace('Layer', '')))
    num_layers  = len(layers)
    num_neurons = hdf_files['experiment'][layers[0]]['Neuron_group_1'].shape[1]

    si_matrix = np.zeros((num_layers, num_neurons))
    pi_matrix = np.zeros((num_layers, num_neurons))

    # Modified frequency sets identical to permutation function
    if split_type == '8-natural' or split_type == '8-naturale' or split_type =='8-zhwiki' or split_type == '8-enwiki':
        si_freq = 4.0 / 8
        pi_freq_list = [4.0 / k for k in range(2, 8)]
    elif split_type == '9-natural' or split_type == '9-naturale':
        si_freq = 4.0 / 9
        pi_freq_list = [4.0 / k for k in range(2, 9)]
    elif split_type == '8-syllable':
        si_freq = 0.5
        pi_freq_list = [1.0]
    else:
        si_freq = 1.0
        pi_freq_list = [2.0]

    freqs = None

    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        layer_name = layers[layer_idx]
        si_neurons = clean_neuron_list(sig_df.loc[layer_idx, 'significant_si_neurons'])
        pi_neurons = clean_neuron_list(sig_df.loc[layer_idx, 'significant_pi_neurons'])

        exp_ffts, ctrl_ffts = {i: [] for i in range(num_neurons)}, {i: [] for i in range(num_neurons)}

        for n in range(num_neurons):
            if n not in si_neurons and n not in pi_neurons:
                continue
            for g in range(10):
                exp = hdf_files['experiment'][layer_name][f'Neuron_group_{g+1}'][:, n]
                ctrl= hdf_files['control-B'][layer_name][f'Neuron_group_{g+1}'][:, n]
                fft_e = rfft(exp);  fft_e[0] = 0
                fft_c = rfft(ctrl); fft_c[0] = 0
                exp_ffts[n].append(np.abs(fft_e))
                ctrl_ffts[n].append(np.abs(fft_c))

            if freqs is None:
                freqs = rfftfreq(len(exp), d=sample_interval)

            idx_si = np.argmin(np.abs(freqs - si_freq))
            idx_list = [np.argmin(np.abs(freqs - p)) for p in pi_freq_list]

            si_val = (np.mean([fft[idx_si] for fft in exp_ffts[n]]) -
                      np.mean([fft[idx_si] for fft in ctrl_ffts[n]]))

            # Modified accept neuron if at least ONE phrase freq diff > 0
            diffs = [(np.mean([fft[i] for fft in exp_ffts[n]]) -
                      np.mean([fft[i] for fft in ctrl_ffts[n]]))
                     for i in idx_list]
            pos_diffs = [d for d in diffs if d > 0]
            pi_val = max(pos_diffs) if pos_diffs else 0

            if n in si_neurons:
                si_matrix[layer_idx, n] = si_val
            if n in pi_neurons:
                pi_matrix[layer_idx, n] = pi_val

    # Z-score analysis, heatmap generation, and CSV output
    si_mean, si_std = np.mean(si_matrix), np.std(si_matrix)
    pi_mean, pi_std = np.mean(pi_matrix), np.std(pi_matrix)
    si_z, pi_z = (si_matrix - si_mean) / si_std, (pi_matrix - pi_mean) / pi_std

    sig_si_counts = np.zeros(num_layers)
    sig_pi_counts = np.zeros(num_layers)
    sig_si_neurons, sig_pi_neurons, shared_neurons = [], [], []

    for layer_idx in range(num_layers):
        si_idx = np.where(si_z[layer_idx] > z_threshold)[0]
        pi_idx = np.where(pi_z[layer_idx] > z_threshold)[0]
        shared_idx = sorted(list(set(si_idx) & set(pi_idx)))
        si_idx = sorted(list(set(si_idx) - set(shared_idx)))
        pi_idx = sorted(list(set(pi_idx) - set(shared_idx)))

        sig_si_neurons.append(si_idx)
        sig_pi_neurons.append(pi_idx)
        shared_neurons.append(shared_idx)
        sig_si_counts[layer_idx] = len(si_idx)
        sig_pi_counts[layer_idx] = len(pi_idx)

    num_bins = (num_neurons + bin_size - 1) // bin_size
    si_bin = np.zeros((num_layers, num_bins))
    pi_bin = np.zeros((num_layers, num_bins))
    highlight_si = np.zeros_like(si_bin, dtype=bool)
    highlight_pi = np.zeros_like(pi_bin, dtype=bool)

    for layer_idx in range(num_layers):
        for bin_idx in range(num_bins):
            s, e = bin_idx * bin_size, min((bin_idx + 1) * bin_size, num_neurons)
            si_bin[layer_idx, bin_idx] = np.mean(si_matrix[layer_idx, s:e])
            pi_bin[layer_idx, bin_idx] = np.mean(pi_matrix[layer_idx, s:e])
            highlight_si[layer_idx, bin_idx] = np.any(np.isin(sig_si_neurons[layer_idx],
                                                              np.arange(s, e)))
            highlight_pi[layer_idx, bin_idx] = np.any(np.isin(sig_pi_neurons[layer_idx],
                                                              np.arange(s, e)))

    cmap = plt.cm.viridis
    yellow = ListedColormap(cmap(np.linspace(0, 1, 512)))
    green  = ListedColormap(cmap(np.linspace(0, .5, 512)))
    norm_si = plt.Normalize(si_bin.min(), si_bin.max())
    norm_pi = plt.Normalize(pi_bin.min(), pi_bin.max())

    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # SI heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(si_bin, cmap=yellow, center=0, mask=~highlight_si, norm=norm_si,
                xticklabels=np.arange(bin_size, num_neurons + 1, bin_size),
                yticklabels=range(1, num_layers + 1), ax=ax,
                cbar_kws={'label': 'Significant $si$'})
    sns.heatmap(si_bin, cmap=green, center=0, mask=highlight_si, norm=norm_si,
                xticklabels=np.arange(bin_size, num_neurons + 1, bin_size),
                yticklabels=range(1, num_layers + 1), ax=ax, cbar=False)
    ax.set_xlabel('Neuron Group'); ax.set_ylabel('Layer')
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'plots',
    #                          f'{split_type}_si_heatmap.png'))
    plt.close()

    # PI heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(pi_bin, cmap=yellow, center=0, mask=~highlight_pi, norm=norm_pi,
                xticklabels=np.arange(bin_size, num_neurons + 1, bin_size),
                yticklabels=range(1, num_layers + 1), ax=ax,
                cbar_kws={'label': 'Significant $pi$'})
    sns.heatmap(pi_bin, cmap=green, center=0, mask=highlight_pi, norm=norm_pi,
                xticklabels=np.arange(bin_size, num_neurons + 1, bin_size),
                yticklabels=range(1, num_layers + 1), ax=ax, cbar=False)
    ax.set_xlabel('Neuron Group'); ax.set_ylabel('Layer')
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'plots',
    #                          f'{split_type}_pi_heatmap.png'))
    plt.close()

    # CSV output
    df_out = pd.DataFrame({
        'Layer': range(1, num_layers + 1),
        'exclusive_si_neurons': [list(map(str, n)) for n in sig_si_neurons],
        'number_of_si_neurons': sig_si_counts.astype(int).tolist(),
        'exclusive_pi_neurons': [list(map(str, n)) for n in sig_pi_neurons],
        'number_of_pi_neurons': sig_pi_counts.astype(int).tolist(),
        'shared_neurons': [list(map(str, n)) for n in shared_neurons],
        'number_of_shared_neurons': [len(n) for n in shared_neurons]
    })
    df_out.to_csv(f'{output_dir}/heatmap/{split_type}_significant_count.csv',
                  index=False)

