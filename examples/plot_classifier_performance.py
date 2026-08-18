#!/usr/bin/env python
"""Evaluate BatBot on ``LABEL/*.wav`` data and plot classifier performance.

Example:
    python examples/plot_classifier_performance.py ./validation --output performance.png

The immediate parent directory of each WAV is its ground-truth label, for
example ``validation/EPFU/recording.wav``.  Labels must use one of the species
codes embedded in the BatBot ONNX model.
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from batbot import classifier

CUSTOM_LABEL = 'NOISE'
GENUS_ORDER_SWAPS = [
    (5, 4),
    (12, 11),
    (11, 10),
    (10, 9),
    (9, 8),
    (30, 29),
]


def apply_genus_order(display, confidences, targets, predicted, custom_index):
    """Keep labels and model outputs aligned while grouping species by genus."""
    display = list(display)
    confidences = confidences.copy()
    targets = targets.copy()
    predicted = predicted.copy()

    for first, second in GENUS_ORDER_SWAPS:
        assert custom_index not in [first, second]
        display[first], display[second] = display[second], display[first]
        confidences[:, [first, second]] = confidences[:, [second, first]]

        first_temporary = 100 + first
        second_temporary = 100 + second

        targets[targets == first] = first_temporary
        targets[targets == second] = second_temporary
        targets[targets == first_temporary] = second
        targets[targets == second_temporary] = first

        predicted[predicted == first] = first_temporary
        predicted[predicted == second] = second_temporary
        predicted[predicted == first_temporary] = second
        predicted[predicted == second_temporary] = first

    return display, confidences, targets, predicted


def shade_regions(display, axis, plot):
    """Shade errors between two-letter taxonomic groups."""
    aliases = {'EUMA': 'ETMA', 'LANO': 'L0N0', 'NYHU': 'NXHU'}
    grouped = [aliases.get(label, label) for label in display]
    regions = sorted({label[:2] for label in grouped})
    errors = 0
    for region in regions:
        indices = [index for index, value in enumerate(grouped) if value[:2] == region]
        minimum, maximum = min(indices), max(indices)
        if minimum > 0:
            errors += plot.confusion_matrix[minimum : maximum + 1, :minimum].sum()
            axis.add_patch(
                patches.Rectangle(
                    (-0.48, minimum - 0.52),
                    minimum,
                    len(indices),
                    edgecolor='none',
                    facecolor=(1.0, 0.0, 0.0, 0.2),
                )
            )
        if maximum < len(display) - 1:
            errors += plot.confusion_matrix[minimum : maximum + 1, maximum + 1 :].sum()
            axis.add_patch(
                patches.Rectangle(
                    (maximum + 1 - 0.48, minimum - 0.52),
                    len(display) - maximum - 1,
                    len(indices),
                    edgecolor='none',
                    facecolor=(1.0, 0.0, 0.0, 0.2),
                )
            )
    return errors / max(1, plot.confusion_matrix.sum())


def run_predictions(paths, cache_path=None, batch_size=classifier.BATCH_SIZE):
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as cache_file:
            cached = json.load(cache_file)
        predictions = cached['results']
        cached_paths = [result['path'] for result in predictions]
        if cached_paths != [str(path) for path in paths]:
            raise ValueError('Prediction cache does not match the discovered WAV files')
        return predictions

    predictions = []
    sessions = {}
    for path in tqdm(paths, desc='Classifying WAV files'):
        predictions.append(
            classifier.classify_wav(
                path,
                batch_size=batch_size,
                sessions=sessions,
            )
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('w') as cache_file:
            json.dump({'results': predictions}, cache_file, indent=2)
    return predictions


def plot_confusion(axis, targets, predicted, labels, display, normalize, title):
    plot = metrics.ConfusionMatrixDisplay.from_predictions(
        targets,
        predicted,
        labels=labels,
        display_labels=display,
        normalize=normalize,
        xticks_rotation='vertical',
        ax=axis,
        values_format='d' if normalize is None else '0.02f',
        text_kw={'fontsize': 5.0},
    )
    for text in plot.text_.ravel():
        if text.get_text() in {'0', '0.00'}:
            text.set_text('')
    shade_regions(display, axis, plot)
    axis.set_title(title, y=1.04)
    return plot


def plot_performance(paths, predictions, output_path):
    classes = classifier.resolve_config().classes
    backward = {label: index for index, label in enumerate(classes)}
    unknown = sorted({path.parent.name for path in paths} - set(classes))
    if unknown:
        raise ValueError('Unknown ground-truth labels: {}'.format(', '.join(unknown)))

    targets = np.asarray([backward[path.parent.name] for path in paths])
    confidences = np.asarray(
        [[prediction['scores'][label] for label in classes] for prediction in predictions]
    )
    predicted = np.argmax(confidences, axis=1)
    custom_index = backward.get(CUSTOM_LABEL)
    display, confidences, targets, predicted = apply_genus_order(
        classes,
        confidences,
        targets,
        predicted,
        custom_index,
    )
    labels = list(range(len(classes)))

    accuracy = metrics.accuracy_score(targets, predicted)
    top_scores = {}
    for top_k in [2, 3, 5]:
        top_scores[top_k] = metrics.top_k_accuracy_score(
            targets,
            confidences,
            k=top_k,
            labels=labels,
        )
    mcc = metrics.matthews_corrcoef(targets, predicted)
    stats = (
        'Top-1 = {:0.2f}% | Top-2 = {:0.2f}% | Top-3 = {:0.2f}% | '
        'Top-5 = {:0.2f}% | MCC = {:0.4f}'
    ).format(
        100 * accuracy,
        100 * top_scores[2],
        100 * top_scores[3],
        100 * top_scores[5],
        mcc,
    )

    dataset_labels = {path.parent.name for path in paths}
    has_noise_examples = CUSTOM_LABEL in dataset_labels and len(dataset_labels) > 1
    if has_noise_examples:
        figure, axes = plt.subplots(2, 3, figsize=(45, 28))
        confusion_axes = axes[0]
    else:
        figure, confusion_axes = plt.subplots(1, 3, figsize=(45, 15))

    absolute_plot = plot_confusion(
        confusion_axes[0],
        targets,
        predicted,
        labels,
        display,
        None,
        f'Confusion Matrix (counts)\n{stats}',
    )
    column_totals = absolute_plot.confusion_matrix.sum(axis=0)
    row_totals = absolute_plot.confusion_matrix.sum(axis=1)
    absolute_plot.ax_.set_xticklabels(
        [f'({value}) {label}' for value, label in zip(column_totals, display)]
    )
    absolute_plot.ax_.set_yticklabels(
        [f'({value}) {label}' for value, label in zip(row_totals, display)]
    )
    plot_confusion(
        confusion_axes[1],
        targets,
        predicted,
        labels,
        display,
        'true',
        f'Confusion Matrix (true-normalized)\n{stats}',
    )
    plot_confusion(
        confusion_axes[2],
        targets,
        predicted,
        labels,
        display,
        'pred',
        f'Confusion Matrix (prediction-normalized)\n{stats}',
    )

    if has_noise_examples:
        noise_index = display.index(CUSTOM_LABEL)
        noise_targets = targets == noise_index
        noise_scores = confidences[:, noise_index]
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            noise_targets, noise_scores
        )
        average_precision = metrics.average_precision_score(noise_targets, noise_scores)
        pr_operating_index = np.argmin(
            np.linalg.norm(np.vstack((1.0 - precision, 1.0 - recall)), axis=0)
        )
        pr_threshold = (
            pr_thresholds[min(pr_operating_index, len(pr_thresholds) - 1)]
            if len(pr_thresholds)
            else 0.5
        )
        metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot(
            ax=axes[1, 0],
            name='MobileNet (AP={:0.3f}, threshold={:0.3f})'.format(
                average_precision, pr_threshold
            ),
        )
        axes[1, 0].set_title('NOISE precision-recall curve')

        false_positive_rate, true_positive_rate, roc_thresholds = metrics.roc_curve(
            noise_targets, noise_scores
        )
        auc = metrics.roc_auc_score(noise_targets, noise_scores)
        roc_operating_index = np.argmin(
            np.linalg.norm(np.vstack((false_positive_rate, 1.0 - true_positive_rate)), axis=0)
        )
        roc_threshold = roc_thresholds[roc_operating_index]
        metrics.RocCurveDisplay(
            fpr=false_positive_rate,
            tpr=true_positive_rate,
            roc_auc=auc,
        ).plot(
            ax=axes[1, 1],
            name=f'MobileNet (AUC={auc:0.3f}, threshold={roc_threshold:0.3f})',
        )
        axes[1, 1].set_title('NOISE ROC curve')

        binary_predictions = noise_scores >= roc_threshold
        binary_confusion = metrics.confusion_matrix(noise_targets, binary_predictions)
        metrics.ConfusionMatrixDisplay(
            binary_confusion,
            display_labels=['species', 'NOISE'],
        ).plot(ax=axes[1, 2], values_format='d')
        axes[1, 2].set_title('NOISE operating-point confusion matrix')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(figure)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='Root directory containing LABEL/*.wav')
    parser.add_argument('--output', type=Path, default=Path('classifier-performance.png'))
    parser.add_argument('--cache', type=Path, default=None, help='Optional prediction JSON cache')
    parser.add_argument('--batch-size', type=int, default=classifier.BATCH_SIZE)
    args = parser.parse_args()

    paths = sorted(path for path in args.data.rglob('*') if path.suffix.lower() == '.wav')
    if not paths:
        parser.error(f'No WAV files found beneath {args.data}')

    predictions = run_predictions(paths, cache_path=args.cache, batch_size=args.batch_size)
    stats = plot_performance(paths, predictions, args.output)
    print(stats)
    print(f'Saved performance plot: {args.output}')


if __name__ == '__main__':
    main()
