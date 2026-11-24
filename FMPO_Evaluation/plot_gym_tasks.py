"""
Plot the Gym MuJoCo evaluation curves (mean ± std over seeds 42-46) from exp_data/.

  python plot_gym_tasks.py              # Figure 3:  <task>_evaluation.pdf (Flow-SDE, ReinFlow, CPPO)
  python plot_gym_tasks.py --ablation   # App. G:  <task>_abl.pdf (adds CPPO without the auxiliary loss)

exp_data/ is produced by FMPO-CPPO/script/export_exp_data.py.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_data")

SEEDS = [42, 43, 44, 45, 46]

AXIS_LABEL_FONTSIZE = 28
TICK_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 28

# Task folder mappings: (flowsde_folder, reinflow_folder, chainppo_folder, max_itr);
# the NoAux ablation reads <chainppo_folder>_noaux.
task_configs = {
    'antv2': ('antv2_FlowSDE', 'antv2_reinflow', 'antv2_chainppo', 600),
    'hopperv2': ('hopperv2_FlowSDE', 'hopperv2_reinflow', 'hopperv2_chainppo', 600),
    'humanoidv3': ('humanoidv3_FlowSDE', 'humanoidv3_reinflow', 'humanoidv3_chainppo', 600),
    'walker2d': ('walker2d_FlowSDE', 'walker2d_reinflow', 'walker2d_chainppo', 600),
}

def load_seed_data(folder_path, data_type='eval', max_itr=None, seeds=SEEDS):
    """Load data from seed CSV files in a folder, restricted to given seeds."""
    all_seeds = []

    for seed in seeds:
        filepath = os.path.join(folder_path, f'seed_{seed}_progress.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing seed record: {filepath}")
        df = pd.read_csv(filepath)
        df = df[df['type'] == data_type]
        if max_itr is not None:
            df = df[df['itr'] <= max_itr]
        if df.empty or df['itr'].duplicated().any():
            raise ValueError(f"Empty or duplicated evaluation grid: {filepath}")
        all_seeds.append(df)

    return all_seeds

def aggregate_seeds(seed_dfs, max_itr=None):
    """Aggregate data across seeds - compute mean and std."""
    if not seed_dfs:
        return None, None, None

    grids = [df['itr'].tolist() for df in seed_dfs]
    if any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("Seed records have different evaluation grids.")
    all_itrs = set()
    for df in seed_dfs:
        all_itrs.update(df['itr'].values)
    all_itrs = sorted(all_itrs)

    rewards_matrix = []
    for df in seed_dfs:
        seed_dict = dict(zip(df['itr'], df['reward']))
        rewards = [seed_dict.get(itr, np.nan) for itr in all_itrs]
        rewards_matrix.append(rewards)

    rewards_matrix = np.array(rewards_matrix)
    mean_rewards = np.nanmean(rewards_matrix, axis=0)
    std_rewards = np.nanstd(rewards_matrix, axis=0)

    return np.array(all_itrs), mean_rewards, std_rewards

def plot_tasks(data_type='eval', ablation=False, output_dir=Path('.')):
    """Plot tasks using data from CSV files."""
    # Validate all inputs before writing any partial figure set.
    for _, (flowsde, reinflow, chain, maximum) in task_configs.items():
        folders = [chain, chain + '_noaux'] if ablation else [flowsde, reinflow, chain]
        for folder in folders:
            load_seed_data(os.path.join(base_dir, folder), data_type, maximum)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    light_red = '#FF6B6B'
    dark_red = '#8B0000'
    chain_color = 'red'  # standard red for CPPO
    noaux_color = '#FFA500'

    for task, (flowsde_folder, reinflow_folder, chainppo_folder, max_itr) in task_configs.items():
        flowsde_dir = os.path.join(base_dir, flowsde_folder)
        reinflow_dir = os.path.join(base_dir, reinflow_folder)
        chainppo_dir = os.path.join(base_dir, chainppo_folder)

        fmpo_seeds = load_seed_data(flowsde_dir, data_type, max_itr) if not ablation else []
        reinflow_seeds = load_seed_data(reinflow_dir, data_type, max_itr) if not ablation else []
        chain_seeds = load_seed_data(chainppo_dir, data_type, max_itr)

        fmpo_itrs, fmpo_mean, fmpo_std = aggregate_seeds(fmpo_seeds, max_itr)
        reinflow_itrs, reinflow_mean, reinflow_std = aggregate_seeds(reinflow_seeds, max_itr)
        chain_itrs, chain_mean, chain_std = aggregate_seeds(chain_seeds, max_itr)
        noaux_seeds = load_seed_data(chainppo_dir + '_noaux', data_type, max_itr) if ablation else []
        noaux_itrs, noaux_mean, noaux_std = aggregate_seeds(noaux_seeds, max_itr)

        print(f"{task}: Flow-SDE seeds={len(fmpo_seeds)}, "
              f"ReinFlow seeds={len(reinflow_seeds)}, "
              f"CPPO seeds={len(chain_seeds)}"
              + (f", CPPO NoAux seeds={len(noaux_seeds)}" if ablation else ""))

        fig, ax = plt.subplots(figsize=(8, 6))

        if fmpo_itrs is not None:
            ax.plot(fmpo_itrs, fmpo_mean, color=light_red, label='Flow-SDE', linewidth=7)
            ax.fill_between(fmpo_itrs, fmpo_mean - fmpo_std, fmpo_mean + fmpo_std,
                           color=light_red, alpha=0.2)

        if reinflow_itrs is not None:
            ax.plot(reinflow_itrs, reinflow_mean, color=dark_red, label='ReinFlow', linewidth=7)
            ax.fill_between(reinflow_itrs, reinflow_mean - reinflow_std, reinflow_mean + reinflow_std,
                           color=dark_red, alpha=0.2)

        if chain_itrs is not None:
            ax.plot(chain_itrs, chain_mean, color=chain_color, label='FMPO-CPPO', linewidth=7)
            ax.fill_between(chain_itrs, chain_mean - chain_std, chain_mean + chain_std,
                           color=chain_color, alpha=0.2)

        if noaux_itrs is not None:
            ax.plot(noaux_itrs, noaux_mean, color=noaux_color, label='NoAux', linewidth=7)
            ax.fill_between(noaux_itrs, noaux_mean - noaux_std, noaux_mean + noaux_std,
                           color=noaux_color, alpha=0.2)

        ax.set_xlabel('Iteration', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Reward', fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.legend(fontsize=LEGEND_FONTSIZE, loc='lower right')

        all_itr_arrays = [a for a in (fmpo_itrs, reinflow_itrs, chain_itrs, noaux_itrs) if a is not None]
        if all_itr_arrays:
            plot_max_itr = max(a.max() for a in all_itr_arrays)
            ax.set_xlim(-plot_max_itr*0.03, plot_max_itr*1.03)

        plt.tight_layout()
        out_file = output_dir / (f'{task}_abl.pdf' if ablation else f'{task}_evaluation.pdf')
        plt.savefig(out_file, bbox_inches='tight')
        print(f'Saved {out_file}')
        plt.close()

    print('\nDone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ablation', action='store_true', help='also plot CPPO without the auxiliary loss (App. G)')
    parser.add_argument('--data-dir', type=Path, default=Path(base_dir), help='Directory containing per-method CSV folders.')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Destination for the four plot panels.')
    args = parser.parse_args()
    base_dir = str(args.data_dir.resolve())
    plot_tasks(data_type='eval', ablation=args.ablation, output_dir=args.output_dir)
