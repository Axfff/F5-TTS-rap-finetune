# # Import pprint for pretty-printing the results in a more readable format
# import pprint
# # Import the SpeechScore class to evaluate speech quality metrics
# from speechscore import SpeechScore
#
# # Main block to ensure the code runs only when executed directly
# if __name__ == '__main__':
#     # Initialize a SpeechScore object with a list of score metrics to be evaluated
#     # Supports any subsets of the list
#
#     # Non-intrusive tests ['NISQA', 'DNSMOS', 'DISTILL_MOS', SRMR'] : No reference audio is required
#
#     mySpeechScore = SpeechScore([
#         'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR',
#         'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS',
#         'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK',
#         'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
#     ])
#     # mySpeechScore = SpeechScore([
#     #     'PESQ', 'NB_PESQ', 'STOI', 'SISDR',
#     #     'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS',
#     #     'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK',
#     #     'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
#     # ])
#
#     # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
#     # Arguments:
#     # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
#     # - window (float): seconds, set None to specify no windowing (process the full audio)
#     # - score_rate (int): specifies the sampling rate at which the metrics should be computed
#     # - return_mean (bool): set True to specify that the mean score for each metric should be returned
#
#
#     print('score for a signle wav file')
#     scores = mySpeechScore(test_path='./speechscore/audios/noisy.wav', reference_path='./speechscore/audios/clean.wav', window=None, score_rate=16000, return_mean=False)
#     #scores = mySpeechScore(test_path='audios/noisy.wav', reference_path=None) # for Non-instrusive tests
#     # Pretty-print the resulting scores in a readable format
#     pprint.pprint(scores)
#
#     print('score for wav directories')
#     scores = mySpeechScore(test_path='./speechscore/audios/noisy/', reference_path='./speechscore/audios/clean/', window=None, score_rate=16000, return_mean=True)
#
#     # Pretty-print the resulting scores in a readable format
#     pprint.pprint(scores)
#
#     # Print only the resulting mean scores in a readable format
#     #pprint.pprint(scores['Mean_Score'])

"""Simple demo script for running SpeechScore metrics and visualizing results."""

import pprint
from collections.abc import Mapping
from pathlib import Path
import numbers

from speechscore import SpeechScore


def _flatten_numeric_scores(values: Mapping, prefix: str = '') -> dict:
    """Flatten nested dicts into {"metric": value} pairs for plotting."""
    flat = {}
    for key, value in values.items():
        composite_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_numeric_scores(value, composite_key))
        elif isinstance(value, numbers.Number):
            flat[composite_key] = float(value)
        else:
            try:
                flat[composite_key] = float(value)
            except (TypeError, ValueError):
                # Skip non-numeric entries (e.g., strings, complex objects).
                continue
    return flat


def plot_scores(values: Mapping, title: str, filename: str) -> None:
    """Create a bar plot for numeric metric values and save it to disk."""
    flat_scores = _flatten_numeric_scores(values)
    if not flat_scores:
        print(f'No numeric values to plot for "{title}".')
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - runtime dependency
        print('matplotlib is not installed. Skipping visualization step.')
        return

    labels = list(flat_scores.keys())
    vals = list(flat_scores.values())
    figure_width = max(6, 0.55 * len(labels))
    figure_height = 4 if len(labels) < 8 else 4 + 0.15 * len(labels)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    bars = ax.bar(range(len(labels)), vals, color='#4C72B0')
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Annotate bars with their numeric value for easier inspection.
    value_range = max(vals) - min(vals)
    offset = 0.02 * value_range if value_range else 0.05
    for bar, val in zip(bars, vals):
        y_position = val + offset if val >= 0 else val - offset
        vertical_alignment = 'bottom' if val >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f'{val:.2f}',
            ha='center',
            va=vertical_alignment,
            fontsize=8,
            color='#222222'
        )

    fig.tight_layout()
    plot_dir = Path(__file__).resolve().parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f'Score plot saved to {output_path}')


# Main block to ensure the code runs only when executed directly
if __name__ == '__main__':
    # Initialize a SpeechScore object with a list of score metrics to be evaluated
    # Supports any subsets of the list

    # Non-intrusive tests ['NISQA', 'DNSMOS', 'DISTILL_MOS', SRMR'] : No reference audio is required

    # mySpeechScore = SpeechScore([
    #     'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR',
    #     'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS',
    #     'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK',
    #     'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
    # ])
    mySpeechScore = SpeechScore([
        'SRMR', 'DNSMOS', 'NISQA', 'DISTILL_MOS'
    ])


    # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
    # Arguments:
    # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
    # - window (float): seconds, set None to specify no windowing (process the full audio)
    # - score_rate (int): specifies the sampling rate at which the metrics should be computed
    # - return_mean (bool): set True to specify that the mean score for each metric should be returned

    print('score for a signle wav file')
    scores = mySpeechScore(test_path='./speechscore/audios/infer_cli_basic.wav', reference_path='./speechscore/audios/clean.wav',
                           window=None, score_rate=16000, return_mean=False)
    # scores = mySpeechScore(test_path='audios/noisy.wav', reference_path=None) # for Non-instrusive tests
    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)
    plot_scores(scores, 'Single Audio Metrics', 'single_audio_scores.png')

    print('score for wav directories')
    scores = mySpeechScore(test_path='./speechscore/audios/noisy/', reference_path='./speechscore/audios/clean/',
                           window=None, score_rate=16000, return_mean=True)

    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)
    mean_scores = scores.get('Mean_Score')
    if mean_scores:
        plot_scores(mean_scores, 'Directory Mean Metrics', 'directory_mean_scores.png')

    # Print only the resulting mean scores in a readable format
    # pprint.pprint(scores['Mean_Score'])

