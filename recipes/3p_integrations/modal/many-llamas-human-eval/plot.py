# ## Plotting HumanEval Results
# This script will calculate pass@k and fail@k for our experiment and plot them.
#
# Run it with:
#    modal run plot

import io
import json
from pathlib import Path
from typing import List, Union
import itertools

import modal

try:
    volume = modal.Volume.lookup("humaneval", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Generate results first with modal run generate --data-dir test --no-dry-run --n 1000 --subsample 100")


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4",
    "pandas==2.2.3",
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
)

app = modal.App("many-llamas-human-eval", image=image)

DATA_DIR = Path("/mnt/humaneval")

with image.imports():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

@app.function(volumes={DATA_DIR: volume})
def render_plots():
    run_dirs = list(sorted((DATA_DIR / "test").glob("run-*")))

    for run_dir in reversed(run_dirs):
        if len(list(run_dir.iterdir())) < 150:
            print(f"skipping incomplete run {run_dir}")
        else:
            break

    all_result_paths = list(run_dir.glob("*.jsonl_results.jsonl"))

    data = []
    for path in all_result_paths:
        data += [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    for element in data:
        del element["completion"]

    df = pd.DataFrame.from_records(data)

    gb = df.groupby("task_id")
    passes = gb["passed"].sum()

    def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
    ) -> np.ndarray:
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

    pass_at_ks = {}

    for k in [1, 10, 100, 1000]:
        pass_at_ks[k] = estimate_pass_at_k(1000, passes, k)

    pass_at_k = {k: np.mean(v) for k, v in pass_at_ks.items()}

    plot_df = pd.DataFrame(
        {"k": pass_at_k.keys(),
         "pass@k": pass_at_k.values()}
    )
    plot_df["fail@k"] = 1 - plot_df["pass@k"]

    sns.set_theme(style='dark')
    plt.style.use("dark_background")

    plt.rcParams['font.sans-serif'] = ["Inter", "Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"]

    sns.despine()

    sns.set_context("talk", rc={"lines.linewidth": 2.5})

    gpt4o_benchmark = 0.902

    # First plot
    plt.figure(figsize=(10, 6))
    fg = sns.lineplot(
        x="k",
        y="pass@k",
        data=plot_df,
        color="#7FEE64",
        linewidth=6,
        alpha=0.9,
        label="Llama 3.2 3B Instruct pass@k"
    )

    initial_lim = fg.axes.get_xlim()
    fg.axes.hlines(
        gpt4o_benchmark, *initial_lim,
        linestyle="--",
        alpha=0.6,
        zorder=-1,
        label="GPT-4o fail@1"
    )
    fg.axes.set_xlim(*initial_lim)
    fg.axes.set_ylabel("")
    fg.axes.set_ylim(0, 1)
    plt.tight_layout(pad=1.2)
    plt.legend()

    # Save the first plot as bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='jpeg')
    plot_1_img_bytes = img_buffer.getvalue()
    plt.close()

    # Second plot
    plt.figure(figsize=(10, 6))
    fg = sns.lineplot(
        x="k",
        y="fail@k",
        data=plot_df,
        color="#7FEE64",
        linewidth=6,
        alpha=0.9,
        label="Llama 3.2 3B Instruct fail@k"
    )

    initial_lim = fg.axes.get_xlim()
    fg.axes.hlines(
        1 - gpt4o_benchmark, *initial_lim,
        linestyle="--",
        alpha=0.6,
        zorder=-1,
        label="GPT-4o fail@1"
    )
    fg.axes.set_xlim(*initial_lim)
    fg.axes.set_ylabel("")
    fg.axes.set_yscale("log")
    fg.axes.set_xscale("log")
    fg.axes.set_xlim(0.5, 2000)
    fg.axes.set_ylim(1e-2, 1e0)
    plt.tight_layout(pad=1.2)
    plt.legend()

    # Save the second plot as bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='jpeg')
    plot_2_img_bytes = img_buffer.getvalue()
    plt.close()

    return [plot_1_img_bytes, plot_2_img_bytes]

@app.local_entrypoint()
def main():
    plots = render_plots.remote()

    assert len(plots) == 2

    with open ("/tmp/plot-pass-k.jpeg", "wb") as f:
        f.write(plots[0])
    
    with open ("/tmp/plot-fail-k.jpeg", "wb") as f:
        f.write(plots[1])

    print("Plots saved to:")
    print("  /tmp/plot-pass-k.jpeg")
    print("  /tmp/plot-fail-k.jpeg")