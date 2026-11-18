'''Figures for the evaluation report: sample pictures, an animation, and charts.'''

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator
from matplotlib.typing import RcKeyType
from PIL import Image, ImageDraw, ImageFont

from digitgan.checkpoints import EpochStats
from digitgan.data import NUM_CLASSES
from digitgan.evaluation import SourceSummary
from digitgan.generation import tile

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
# Real data and the baseline are references, so they stay neutral
NEUTRAL = '#a8a69f'
# Categorical slots in fixed order. Models take them in configuration order, so a model keeps
# its colour in every figure. The first three stay distinguishable for colour-blind readers
# in every pairing.
CATEGORICAL = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
GENERATOR_COLOUR = '#6250d6'
DISCRIMINATOR_COLOUR = '#e34948'
SEQUENTIAL_BLUE = ('#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b')

# Light gaps between digits in sample pictures, close to the chart surface colour
GAP_VALUE = 252
# 32 grey levels look the same as 256 at this size and halve the size of the animation
ANIMATION_GREY_LEVELS = 32
DPI = 200

_STYLE: dict[RcKeyType, Any] = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titlelocation': 'left',
    'axes.titlecolor': INK,
    'axes.labelcolor': INK_SECONDARY,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'grid.linestyle': '-',
    'legend.frameon': False,
    'legend.labelcolor': INK_SECONDARY,
    'lines.linewidth': 1.6,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
}


def model_colours(model_keys: Sequence[str]) -> dict[str, str]:
    '''Assign categorical colours to models in configuration order.'''
    return {key: CATEGORICAL[index % len(CATEGORICAL)] for index, key in enumerate(model_keys)}


def save_picture(images: np.ndarray, path: Path, scale: int = 2) -> Path:
    '''Save 8-bit images of shape (rows, columns, 28, 28) as one PNG with light gaps.'''
    picture = Image.fromarray(tile(images, padding=2, fill=GAP_VALUE), mode='L')
    picture = picture.resize(
        (picture.width * scale, picture.height * scale), Image.Resampling.NEAREST
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path, optimize=True)
    return path


def save_training_animation(
    frames: np.ndarray, path: Path, epochs: Sequence[int], rows: int = 5, scale: int = 2
) -> Path:
    '''Save selected progress frames as a looping GIF with an epoch caption.

    Args:
        frames: 8-bit frames of shape (epochs + 1, 100, 28, 28); frame 0 is before training.
        path: Output file.
        epochs: Which epochs to show, in order.
        rows: How many of the ten rows of each frame to show.
        scale: Pixel enlargement factor.
    '''
    font = ImageFont.load_default(size=13 * scale // 2)
    step = 255 / (ANIMATION_GREY_LEVELS - 1)
    pictures = []
    for epoch in epochs:
        shown = frames[epoch, : rows * NUM_CLASSES].reshape(rows, NUM_CLASSES, *frames.shape[2:])
        posterised = (np.round(shown / step) * step).astype(np.uint8)
        grid = tile(posterised, 2, GAP_VALUE)
        picture = Image.fromarray(grid, mode='L')
        picture = picture.resize(
            (picture.width * scale, picture.height * scale), Image.Resampling.NEAREST
        )
        canvas = Image.new('L', (picture.width, picture.height + 22 * scale // 2), GAP_VALUE)
        canvas.paste(picture, (0, 0))
        caption = 'Before training' if epoch == 0 else f'Epoch {epoch}'
        ImageDraw.Draw(canvas).text((4, picture.height + 3), caption, fill=40, font=font)
        pictures.append(canvas)

    # Linger on the last frame so the final result is easy to see
    durations = [300] * (len(pictures) - 1) + [2500]
    path.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(
        path, save_all=True, append_images=pictures[1:], duration=durations, loop=0, optimize=True
    )
    return path


def plot_model_comparison(rows: Sequence[tuple[str, str, np.ndarray]], path: Path) -> Path:
    '''Show ten samples from each source, one source per row.

    Args:
        rows: Name, caption, and 8-bit images of shape (10, 28, 28) for each source.
        path: Output file.
    '''
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(len(rows), 1, figsize=(7.2, 0.66 * len(rows) + 0.1))
        for axis, (name, caption, images) in zip(np.atleast_1d(axes), rows, strict=True):
            axis.imshow(tile(images[np.newaxis], 2, GAP_VALUE), cmap='gray', vmin=0, vmax=255)
            axis.set_axis_off()
            axis.text(
                -0.02, 0.58, name, transform=axis.transAxes, ha='right', color=INK, fontsize=9.5
            )
            axis.text(
                -0.02,
                0.28,
                caption,
                transform=axis.transAxes,
                ha='right',
                color=INK_MUTED,
                fontsize=8,
            )
        figure.subplots_adjust(left=0.24, right=0.995, top=0.99, bottom=0.01, hspace=0.08)
        return _save(figure, path)


def plot_pixel_mse(
    summaries: Sequence[SourceSummary], colours: Mapping[str, str], path: Path
) -> Path:
    '''Compare pixel MSE like for like: sources paired with the same digit, then with any digit.'''
    groups = [
        ('Paired with a test image of the same digit', [s for s in summaries if s.conditional]),
        ('Paired with a test image of any digit', [s for s in summaries if not s.conditional]),
    ]
    groups = [(title, members) for title, members in groups if members]
    limit = 1.18 * max(s.pixel_mse.mean + s.pixel_mse.std for s in summaries)

    with plt.rc_context(_STYLE):
        heights = [len(members) for _, members in groups]
        figure, axes = plt.subplots(
            len(groups),
            1,
            figsize=(7.2, 0.42 * sum(heights) + 0.75 * len(groups) + 0.3),
            gridspec_kw={'height_ratios': heights},
            sharex=True,
        )
        for axis, (title, members) in zip(np.atleast_1d(axes), groups, strict=True):
            positions = np.arange(len(members))[::-1]
            means = [s.pixel_mse.mean for s in members]
            bar_colours = [colours.get(s.key, NEUTRAL) for s in members]
            axis.barh(positions, means, height=0.56, color=bar_colours)
            for position, summary in zip(positions, members, strict=True):
                end = summary.pixel_mse.mean
                # Error bars show the spread across seeds, so single runs get none
                if len(summary.seeds) > 1:
                    axis.errorbar(
                        summary.pixel_mse.mean,
                        position,
                        xerr=summary.pixel_mse.std,
                        fmt='none',
                        ecolor=INK_SECONDARY,
                        elinewidth=0.8,
                        capsize=2,
                    )
                    end += summary.pixel_mse.std
                axis.text(
                    end + limit * 0.012,
                    position,
                    f'{summary.pixel_mse.mean:.3f}',
                    va='center',
                    color=INK,
                    fontsize=8.5,
                )
            axis.set_yticks(positions, [s.display_name for s in members])
            axis.tick_params(axis='y', length=0)
            axis.set_title(title, color=INK_SECONDARY, fontsize=9)
            axis.set_xlim(0, limit)
            axis.grid(axis='x')
            axis.set_axisbelow(True)
        np.atleast_1d(axes)[-1].set_xlabel('Pixel MSE (lower is closer to the test image)')
        figure.tight_layout()
        return _save(figure, path)


def plot_fid_during_training(
    histories: Mapping[str, Sequence[Sequence[EpochStats]]],
    names: Mapping[str, str],
    colours: Mapping[str, str],
    real_fid: float,
    path: Path,
) -> Path:
    '''Plot tracked FID against epoch for each model: the mean over seeds, with the range shaded.'''
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 3.6))
        end_labels = []
        for key, runs in histories.items():
            epochs, values = _tracked(runs, 'fid')
            mean = values.mean(axis=0)
            axis.fill_between(
                epochs,
                values.min(axis=0),
                values.max(axis=0),
                color=colours[key],
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(
                epochs,
                mean,
                color=colours[key],
                label=names[key],
                marker='o',
                markersize=4,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            label = axis.annotate(
                f'{mean[-1]:.1f}',
                (epochs[-1], mean[-1]),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                color=INK,
                fontsize=8.5,
            )
            end_labels.append(label)
        axis.axhline(real_fid, color=INK_MUTED, linewidth=1)
        axis.annotate(
            f'Real images {real_fid:.1f}',
            (0, real_fid),
            xytext=(2, 4),
            textcoords='offset points',
            color=INK_MUTED,
            fontsize=8,
        )
        axis.set_yscale('log')
        axis.set_xlim(0, None)
        axis.set_xlabel('Epoch')
        _set_epoch_ticks(axis)
        axis.set_ylabel('FID (log scale, lower is better)')
        axis.grid(axis='y', which='major')
        axis.set_axisbelow(True)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper right', ncols=len(histories))
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        _separate_labels(figure, axis, end_labels)
        return _save(figure, path)


def plot_training_losses(
    histories: Mapping[str, Sequence[Sequence[EpochStats]]], names: Mapping[str, str], path: Path
) -> Path:
    '''Plot generator and discriminator losses per model, as small multiples on one scale.'''
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            1, len(histories), figsize=(7.2, 2.7), sharey=True, squeeze=False
        )
        for axis, (key, runs) in zip(axes[0], histories.items(), strict=True):
            for field, colour, label in (
                ('generator_loss', GENERATOR_COLOUR, 'Generator'),
                ('discriminator_loss', DISCRIMINATOR_COLOUR, 'Discriminator'),
            ):
                epochs, values = _tracked(runs, field, every_epoch=True)
                axis.fill_between(
                    epochs,
                    values.min(axis=0),
                    values.max(axis=0),
                    color=colour,
                    alpha=0.12,
                    linewidth=0,
                )
                axis.plot(epochs, values.mean(axis=0), color=colour, label=label)
            axis.set_title(names[key])
            axis.set_xlabel('Epoch')
            _set_epoch_ticks(axis)
            axis.grid(axis='y')
            axis.set_axisbelow(True)
        axes[0][0].set_ylabel('Loss')
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper right', ncols=2)
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        return _save(figure, path)


def plot_class_distribution(
    summaries: Sequence[tuple[str, str, Sequence[float]]], colours: Mapping[str, str], path: Path
) -> Path:
    '''Grouped bars of how often the classifier sees each digit, per unconditional source.

    Args:
        summaries: Key, display name, and the share of each digit for each source.
        colours: Colour per key. Keys without one are drawn in the neutral reference colour.
        path: Output file.
    '''
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 3.0))
        width = 0.8 / len(summaries)
        digits = np.arange(NUM_CLASSES)
        for index, (key, name, shares) in enumerate(summaries):
            offset = (index - (len(summaries) - 1) / 2) * width
            axis.bar(
                digits + offset,
                100 * np.asarray(shares),
                width=width * 0.9,
                color=colours.get(key, NEUTRAL),
                label=name,
            )
        axis.axhline(100 / NUM_CLASSES, color=INK_MUTED, linewidth=1, label='Equal share')
        axis.set_xticks(digits)
        axis.set_xlabel('Digit predicted by the classifier')
        axis.set_ylabel('Share of samples (%)')
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper right', ncols=len(labels))
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        return _save(figure, path)


def plot_confusion(confusion: np.ndarray, title: str, path: Path) -> Path:
    '''Heatmap of requested digit against the classifier's prediction, as row percentages.'''
    percentages = 100 * confusion / confusion.sum(axis=1, keepdims=True)
    colour_map = LinearSegmentedColormap.from_list('sequential_blue', ['#ffffff', *SEQUENTIAL_BLUE])
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(4.6, 4.2))
        axis.imshow(percentages, cmap=colour_map, vmin=0, vmax=100)
        for row in range(NUM_CLASSES):
            for column in range(NUM_CLASSES):
                value = percentages[row, column]
                # Label only cells that carry information; empty cells stay blank
                if value >= 1:
                    axis.text(
                        column,
                        row,
                        f'{value:.0f}',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='#ffffff' if value >= 50 else INK,
                    )
        axis.set_xticks(range(NUM_CLASSES))
        axis.set_yticks(range(NUM_CLASSES))
        axis.tick_params(length=0)
        axis.set_xlabel('Classified as')
        axis.set_ylabel('Requested digit')
        axis.set_title(title)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.tight_layout()
        return _save(figure, path)


def _set_epoch_ticks(axis: Axes) -> None:
    '''Tick the epoch axis at round whole numbers such as 0, 10, 20.'''
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))


def _separate_labels(
    figure: Figure, axis: Axes, labels: Sequence[Annotation], gap: float = 11.0
) -> None:
    '''Move value labels upwards where needed so that they are at least gap points apart.

    Call it after the layout is final, because it works in display coordinates.
    '''
    figure.canvas.draw()
    points_per_pixel = 72 / figure.dpi
    anchored = [
        (axis.transData.transform(label.xy)[1] * points_per_pixel, label) for label in labels
    ]
    previous = -np.inf
    for anchor, label in sorted(anchored, key=lambda item: item[0]):
        placed = max(anchor, previous + gap)
        label.xyann = (label.xyann[0], placed - anchor)
        previous = placed


def _tracked(
    runs: Sequence[Sequence[EpochStats]], field: str, every_epoch: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    '''Return the epochs and a (runs, epochs) array of one recorded field.'''
    first = [stats for stats in runs[0] if every_epoch or getattr(stats, field) is not None]
    epochs = np.array([stats.epoch for stats in first])
    by_epoch = [{stats.epoch: getattr(stats, field) for stats in run} for run in runs]
    values = np.array([[run[epoch] for epoch in epochs] for run in by_epoch], dtype=float)
    return epochs, values


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path
