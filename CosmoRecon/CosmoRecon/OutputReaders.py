"""
Module for reading, analysing and plotting reconstructed cosmological fields.

Classes
-------
OutputReader
    Loads density fields from different reconstruction methods (real-space,
    redshift-space, linear-theory, neural-network, combined) and computes
    power-spectrum multipoles, discrepancies, residuals and chi-squared
    statistics.

Plotter
    Publication-quality plots for 3D power-spectrum multipoles and residuals,
    supporting comparison across multiple ``OutputReader`` instances.

Plotter2D
    Similar plotting utilities for 2D reconstruction results (input / target /
    prediction power spectra and spatial maps).
"""

import logging
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from glob import glob
from tqdm import tqdm
from pypower.fft_power import MeshFFTPower
from pmesh.pm import RealField, ParticleMesh

logger = logging.getLogger(__name__)

# Default TeXLive path -- override with COSMORECON_TEXLIVE_PATH env var
_DEFAULT_TEXLIVE = os.environ.get(
    'COSMORECON_TEXLIVE_PATH',
    r'/farmdisk1/cosmology/Libraries/texlive/2024/bin/x86_64-linux',
)


# ---------------------------------------------------------------------------
# LaTeX environment setup
# ---------------------------------------------------------------------------

def set_latex_env(texlive_path=None):
    """Configure Matplotlib for publication-quality LaTeX rendering.

    Parameters
    ----------
    texlive_path : str or None
        Directory containing the ``latex`` binary.  If ``None``, the function
        first checks whether ``latex`` is already on ``PATH``; if not, it
        falls back to ``_DEFAULT_TEXLIVE``.
    """
    if texlive_path is None:
        if shutil.which('latex') is None:
            texlive_path = _DEFAULT_TEXLIVE
            logger.info("latex not on PATH; using default: %s", texlive_path)
    if texlive_path is not None:
        os.environ['PATH'] = texlive_path + os.pathsep + os.environ.get('PATH', '')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Computer Modern Roman')
    rcParams['text.latex.preamble'] = (
        r'\usepackage{amsmath}'
        r'\newcommand{\sfont}[1]{{\scriptscriptstyle\rm #1}}'
    )
    rcParams.update({
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'axes.labelsize': 14,
        'legend.fontsize': 13,
        'font.size': 13,
        'figure.figsize': (6.6, 4.8),
        'legend.handletextpad': 0.8,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.5,
        'legend.borderaxespad': 0.5,
    })


# ---------------------------------------------------------------------------
# OutputReader
# ---------------------------------------------------------------------------

class OutputReader:
    """Load fields and compute power-spectrum statistics for different
    reconstruction methods.

    Parameters
    ----------
    real_space, redshift_space, lt_rec, nn_rec, lt_nn_rec : str or None
        Glob-compatible paths to density field ``.npy`` files for each method.
    name : str
        Human-readable label used in plots and logs.
    """

    def __init__(self, real_space=None, redshift_space=None, lt_rec=None,
                 nn_rec=None, lt_nn_rec=None, name="OutputReader"):
        self.path_dict = {
            'real_space': real_space,
            'redshift_space': redshift_space,
            'lt_rec': lt_rec,
            'nn_rec': nn_rec,
            'lt_nn_rec': lt_nn_rec,
        }
        self._fields_dict = {}
        self._pk_multipoles = {}
        self._pk_residuals = {}
        self._pk_discrepancies = {}
        self.name = name

    # -- properties ----------------------------------------------------------

    @property
    def fields_dict(self):
        return self._fields_dict

    @property
    def pk_multipoles(self):
        return self._pk_multipoles

    @pk_multipoles.setter
    def pk_multipoles(self, value):
        if not isinstance(value, dict):
            raise ValueError("pk_multipoles must be a dict.")
        self._pk_multipoles = value

    @property
    def pk_residuals(self):
        return self._pk_residuals

    @pk_residuals.setter
    def pk_residuals(self, value):
        if not isinstance(value, dict):
            raise ValueError("pk_residuals must be a dict.")
        self._pk_residuals = value

    @property
    def pk_discrepancies(self):
        return self._pk_discrepancies

    @pk_discrepancies.setter
    def pk_discrepancies(self, value):
        if not isinstance(value, dict):
            raise ValueError("pk_discrepancies must be a dict.")
        self._pk_discrepancies = value

    # -- field loading -------------------------------------------------------

    def load_fields(self, idxs=None):
        """Load fields for all modes that have a defined path."""
        for mode, path in self.path_dict.items():
            if path is not None:
                self._load_fields(mode, idxs)
            else:
                logger.debug("No path for mode '%s', skipping.", mode)

    def _load_fields(self, mode, idxs=None):
        if mode not in self.path_dict or self.path_dict[mode] is None:
            raise ValueError(f"No path defined for mode '{mode}'.")

        all_files = sorted(glob(self.path_dict[mode]))
        self._fields_dict[mode] = {}

        if mode in ('real_space', 'redshift_space', 'lt_rec'):
            files = [all_files[i] for i in idxs] if idxs is not None else all_files
            for i, fpath in enumerate(files):
                self._fields_dict[mode][i] = np.load(fpath)

        elif mode in ('nn_rec', 'lt_nn_rec'):
            if len(all_files) == 1:
                all_mocks = np.load(all_files[0])
                all_mocks = self._format_nn_rec_field(all_mocks)
                for i in range(all_mocks.shape[0]):
                    self._fields_dict[mode][i] = all_mocks[i]
            else:
                files = [all_files[i] for i in idxs] if idxs is not None else all_files
                for i, fpath in enumerate(files):
                    self._fields_dict[mode][i] = np.load(fpath)

    @staticmethod
    def _format_nn_rec_field(field):
        """Remove trailing channel dimension if present."""
        if field.ndim == 5 and field.shape[-1] == 1:
            field = np.squeeze(field, axis=-1)
        return field

    # -- power-spectrum computation ------------------------------------------

    def _compute_pk_multipoles(self, mock_field, grid_size, box_size, box_centre):
        """Compute P(k) multipoles (ell = 0, 2, 4) for a single mock."""
        if mock_field.shape != (grid_size, grid_size, grid_size):
            raise ValueError(
                f"Expected shape ({grid_size},)*3, got {mock_field.shape}"
            )

        k_fundamental = 2 * np.pi / box_size
        k_nyquist = k_fundamental * grid_size / 2
        edges = np.arange(k_fundamental, k_nyquist, k_fundamental)

        pm = ParticleMesh(
            Nmesh=[grid_size, grid_size, grid_size], BoxSize=box_size,
        )
        field_obj = RealField(pm)
        field_obj.value[:] = mock_field

        pk_obj = MeshFFTPower(
            field_obj, edges=edges, ells=(0, 2, 4),
            los='z', boxcenter=box_centre,
        )
        k = pk_obj.poles.k
        pk_poles = np.array([
            pk_obj.poles.power[0].real,
            pk_obj.poles.power[1].real,
            pk_obj.poles.power[2].real,
        ])
        return k, pk_poles

    def compute_all_pk_multipoles(self, modes, grid_size, box_size, box_centre):
        """Compute P(k) multipoles for every loaded mock in the given modes."""
        for mode in modes:
            if mode not in self._fields_dict:
                raise ValueError(f"Fields for '{mode}' not loaded.")
            if mode not in self._pk_multipoles:
                self._pk_multipoles[mode] = {}

            logger.info("Computing pk_multipoles for mode: %s", mode)
            for i, data in tqdm(self._fields_dict[mode].items()):
                k, pk_poles = self._compute_pk_multipoles(
                    data, grid_size, box_size, box_centre,
                )
                self._pk_multipoles[mode][i] = pk_poles

        # k values are the same for all mocks/modes
        self.k_values = k

    def compute_mean_pk_multipoles(self):
        """Compute the mean multipoles across mocks for each mode."""
        self.mean_pk_multipoles = {}
        for mode, multipole_dict in self._pk_multipoles.items():
            all_poles = np.array(list(multipole_dict.values()))
            self.mean_pk_multipoles[mode] = np.mean(all_poles, axis=0)

    def compute_pk_discrepancies(self):
        """Compute discrepancies w.r.t. real-space multipoles."""
        if 'real_space' not in self._pk_multipoles:
            raise ValueError("Real-space multipoles not computed.")

        for mode in self._pk_multipoles:
            if mode == 'real_space':
                continue
            if mode not in self._pk_discrepancies:
                self._pk_discrepancies[mode] = {}
            for i in self._pk_multipoles[mode]:
                real = self._pk_multipoles['real_space'][i]
                obs = self._pk_multipoles[mode][i]
                self._pk_discrepancies[mode][i] = obs - real

    def compute_pk_residuals(self):
        """Compute normalised residuals (mean / std) from discrepancies.

        k-bins where the standard deviation is zero are set to 0 in the
        residuals (rather than producing inf/nan).
        """
        if not self._pk_discrepancies:
            raise ValueError("Compute discrepancies first.")

        for mode, discr_dict in self._pk_discrepancies.items():
            discr_array = np.array(list(discr_dict.values()))
            mean_discr = np.mean(discr_array, axis=0)
            std_discr = np.std(discr_array, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                residuals = np.where(
                    std_discr != 0, mean_discr / std_discr, 0.0,
                )
            n_zero = np.sum(std_discr == 0)
            if n_zero > 0:
                logger.warning(
                    "Mode '%s': %d k-bins with std=0 set to residual=0",
                    mode, n_zero,
                )
            self._pk_residuals[mode] = residuals

    def compute_chi_squares(self):
        """Compute chi-squared for each multipole from normalised residuals."""
        if not self._pk_residuals:
            raise ValueError("Compute residuals first.")

        self.chi_squares = {}
        for mode, res in self._pk_residuals.items():
            self.chi_squares[mode] = {
                'mon': np.sum(res[0] ** 2),
                'quad': np.sum(res[1] ** 2),
                'hex': np.sum(res[2] ** 2),
            }

    def compute_all_stats(self, modes, grid_size, box_size, box_centre):
        """Run the full analysis pipeline: multipoles -> discrepancies ->
        residuals -> chi-squared."""
        self.compute_all_pk_multipoles(modes, grid_size, box_size, box_centre)
        self.compute_mean_pk_multipoles()
        self.compute_pk_discrepancies()
        self.compute_pk_residuals()
        self.compute_chi_squares()


# ---------------------------------------------------------------------------
# Plotter (3D multipoles)
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = {
    'real_space': 'black',
    'redshift_space': 'tab:orange',
    'lt_rec': 'grey',
    'nn_rec': 'tab:blue',
    'lt_nn_rec': 'tab:red',
}

_DEFAULT_LEGEND = {
    'real_space': 'Real Space',
    'redshift_space': 'Redshift Space',
    'lt_rec': 'LT',
    'nn_rec': 'NN',
    'lt_nn_rec': 'LT+NN',
}


class Plotter:
    """Publication-quality plotter for 3D power-spectrum multipoles.

    Supports overlaying results from multiple ``OutputReader`` instances
    using distinct line styles.

    Parameters
    ----------
    readers : OutputReader or list[OutputReader]
        One or more reader objects with pre-computed statistics.
    colors_dict : dict or None
        Mode -> colour mapping.
    linestyles : list or None
        Line styles for each reader.
    """

    def __init__(self, readers, colors_dict=None, linestyles=None):
        if not isinstance(readers, (list, tuple)):
            readers = [readers]
        self.readers = readers

        self.colors_dict = colors_dict or dict(_DEFAULT_COLORS)
        default_ls = ['-', '--', ':', '-.']
        self.linestyles = linestyles or default_ls[:len(readers)]

        self.pkylabel = [
            r"$P_0(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$",
            r"$P_2(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$",
            r"$P_4(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$",
        ]
        self.pkxlabel = r"$k\, [h~\mathrm{Mpc}^{-1}]$"
        self.legend_labels_dict = dict(_DEFAULT_LEGEND)

    # -- internal helpers ----------------------------------------------------

    def _make_legend_handles(self, modes, legend_labels_dict=None):
        labels = legend_labels_dict or self.legend_labels_dict
        handles = []
        for mode in ['real_space'] + modes:
            handle = plt.Line2D(
                [], [], color=self.colors_dict.get(mode),
                lw=2, label=labels.get(mode, mode),
            )
            handles.append(handle)

        ls_handles = []
        if len(self.readers) > 1:
            ls_handles = [
                plt.Line2D([], [], color='dimgray', lw=2, linestyle=ls,
                           label=reader.name)
                for reader, ls in zip(self.readers, self.linestyles)
            ]
        return handles, ls_handles

    def _compose_label(self, mode, reader_idx):
        base = self.legend_labels_dict.get(mode, mode)
        if len(self.readers) > 1:
            return f"{base} (R{reader_idx + 1})"
        return base

    # -- public plots --------------------------------------------------------

    def plot_pk_residuals(self, modes=None, title=None, y_label='pk_residuals',
                          figsize=(14, 4), RESPANELYLIM=5.0,
                          legend_labels_dict=None, share_col=True,
                          LABELSIZE=11):
        """Plot normalised residuals for ell = 0, 2, 4."""
        if modes is None:
            modes = list(self.readers[0].pk_residuals.keys())

        k = self.readers[0].k_values
        fig, axs = plt.subplots(
            1, 3, figsize=figsize, sharex='col' if share_col else False,
        )

        for ell in range(3):
            for i, reader in enumerate(self.readers):
                ls = self.linestyles[i]
                for mode in modes:
                    axs[ell].plot(
                        k, reader.pk_residuals[mode][ell],
                        label=self._compose_label(mode, i),
                        color=self.colors_dict.get(mode),
                        linestyle=ls, linewidth=2,
                    )
            axs[ell].set_xlim(np.min(k), np.max(k))
            axs[ell].set_xscale('log')
            axs[ell].set_xlabel(self.pkxlabel)
            axs[ell].set_ylabel(y_label)
            axs[ell].axhline(0, ls='--', color='black', lw=1)
            axs[ell].fill_between(k, -1, 1, alpha=0.2, color='grey')
            axs[ell].fill_between(k, -2, 2, alpha=0.1, color='grey')
            axs[ell].set_ylim(-RESPANELYLIM, RESPANELYLIM)
            axs[ell].set_yticks(np.arange(-RESPANELYLIM, RESPANELYLIM + 1, 2))

        handles_c, handles_ls = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(
            handles=handles_c + handles_ls, loc='lower center',
            ncol=max(len(modes) + 1, len(self.readers)),
            frameon=False, fontsize=LABELSIZE,
        )
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig, axs

    def plot_pk_multipoles_and_residuals(
        self, modes=None, multipoles=(0, 2, 4), title=None,
        legend_labels_dict=None, figsize=(11.7, 5), RESPANELYLIM=5.0,
        UPPERPANELYLIM=None, width_ratios=None, height_ratios=None,
        share_col=True, hspace=None, wspace=0.3, LABELSIZE=11,
    ):
        """Two-row plot: multipoles on top, residuals on the bottom."""
        if modes is None:
            modes = [
                m for m in self.readers[0].pk_multipoles.keys()
                if m != 'real_space'
            ]

        if share_col:
            hspace = 0

        k = self.readers[0].k_values
        L = len(multipoles)

        w_ratios = width_ratios or [1] * L
        h_ratios = height_ratios or [2, 1]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, L, height_ratios=h_ratios, width_ratios=w_ratios,
            hspace=hspace, wspace=wspace,
        )

        # Upper panels: P(k) multipoles
        axs = np.empty(L, dtype=object)
        for ell in range(L):
            axs[ell] = fig.add_subplot(gs[0, ell])
            for i, reader in enumerate(self.readers):
                ls = self.linestyles[i]
                for mode in modes + ['real_space']:
                    mean_poles = reader.mean_pk_multipoles[mode]
                    axs[ell].plot(
                        k, mean_poles[ell],
                        color=self.colors_dict.get(mode),
                        linestyle=ls, linewidth=2,
                    )
            axs[ell].tick_params(
                axis='both', which='both', direction='in', labelsize=LABELSIZE,
            )
            axs[ell].set_xscale('log')
            axs[ell].set_xlim(np.min(k), np.max(k))
            axs[ell].set_xlabel(self.pkxlabel)
            axs[ell].set_ylabel(self.pkylabel[ell])
            if UPPERPANELYLIM is not None:
                axs[ell].set_ylim(UPPERPANELYLIM)
            if ell == 0:
                axs[ell].set_yscale('log')
            if ell > 0:
                axs[ell].axhline(0, ls='--', color='black', lw=1)

        # Lower panels: residuals
        res = np.empty(L, dtype=object)
        for ell in range(L):
            res[ell] = fig.add_subplot(
                gs[1, ell], sharex=axs[ell] if share_col else None,
            )
            if share_col:
                axs[ell].tick_params(labelbottom=False)

            for i, reader in enumerate(self.readers):
                ls = self.linestyles[i]
                for mode in modes:
                    res[ell].plot(
                        k, reader.pk_residuals[mode][ell],
                        color=self.colors_dict.get(mode),
                        linestyle=ls, linewidth=2,
                    )
            res[ell].set_xlim(np.min(k), np.max(k))
            res[ell].set_xlabel(self.pkxlabel)
            res[ell].axhline(0, ls='--', color='black', lw=1)
            res[ell].fill_between(k, -1, 1, alpha=0.2, color='grey')
            res[ell].fill_between(k, -2, 2, alpha=0.1, color='grey')
            res[ell].set_ylim(-RESPANELYLIM, RESPANELYLIM)
            res[ell].set_yticks(
                np.arange(-RESPANELYLIM, RESPANELYLIM + 1, 2),
            )
            res[ell].set_xscale('log')
            res[ell].set_ylabel('residuals')

        handles_c, handles_ls = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(
            handles=handles_c, loc='lower center', ncol=len(modes) + 1,
            bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=LABELSIZE,
        )
        if handles_ls:
            fig.legend(
                handles=handles_ls, loc='lower center', ncol=len(self.readers),
                bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=LABELSIZE,
            )
        fig.suptitle(title)
        return fig


# ---------------------------------------------------------------------------
# Plotter2D (2D reconstruction results)
# ---------------------------------------------------------------------------

class Plotter2D:
    """Plotting utilities for 2D reconstruction experiments.

    Parameters
    ----------
    readers : OutputReader or list
        Reader(s) with 2D power-spectrum results.
    colors_dict, linestyles : dict, list or None
        Customisable appearance.
    """

    def __init__(self, readers, colors_dict=None, linestyles=None):
        if not isinstance(readers, (list, tuple)):
            readers = [readers]
        self.readers = readers

        self.colors_dict = colors_dict or {
            'input': 'grey',
            'target': 'black',
            'pred': 'tab:blue',
            'residual_pred': 'tab:red',
        }
        default_ls = ['-', '--', ':', '-.']
        self.linestyles = linestyles or default_ls[:len(readers)]

        self.pkylabel = r"$P(k)\, [\mathrm{deg}^{2}]$"
        self.pkxlabel = r"$k\, [h~\mathrm{Mpc}^{-1}]$"
        self.legend_labels_dict = {
            'input': 'Input (Masked)',
            'target': 'Target (Truth)',
            'pred': 'UNet Prediction',
            'residual_pred': 'Residual Component',
        }

    def _make_legend_handles(self, modes):
        handles = [
            plt.Line2D(
                [], [], color=self.colors_dict.get(m, 'black'),
                lw=2, label=self.legend_labels_dict.get(m, m),
            )
            for m in modes
        ]
        ls_handles = []
        if len(self.readers) > 1:
            ls_handles = [
                plt.Line2D([], [], color='dimgray', lw=2, linestyle=ls,
                           label=r.name)
                for r, ls in zip(self.readers, self.linestyles)
            ]
        return handles, ls_handles

    def plot_pk_and_ratio(self, title=None, figsize=(8, 8),
                          k_range=None, ratio_ylim=(0.8, 1.2)):
        """P(k) on top and P_pred / P_target ratio on the bottom."""
        modes = ['target', 'input', 'pred']
        k = self.readers[0].k_values

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

        for i, reader in enumerate(self.readers):
            ls = self.linestyles[i]
            for mode in modes:
                if mode in reader.pk_multipoles:
                    pk_vals = np.array(list(reader.pk_multipoles[mode].values()))
                    pk_mean = np.mean(pk_vals, axis=0)
                    ax0.plot(
                        k, pk_mean[0], color=self.colors_dict[mode],
                        ls=ls, lw=2, label=f"{mode}_{i}",
                    )
            if 'pred' in reader.pk_multipoles and 'target' in reader.pk_multipoles:
                pred_vals = np.array(list(reader.pk_multipoles['pred'].values()))
                target_vals = np.array(list(reader.pk_multipoles['target'].values()))
                ratio = pred_vals[:, 0, :] / (target_vals[:, 0, :] + 1e-30)
                mean_ratio = np.mean(ratio, axis=0)
                std_ratio = np.std(ratio, axis=0)
                ax1.plot(k, mean_ratio, color=self.colors_dict['pred'], ls=ls, lw=2)
                ax1.fill_between(
                    k, mean_ratio - std_ratio, mean_ratio + std_ratio,
                    color=self.colors_dict['pred'], alpha=0.2,
                )

        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_ylabel(self.pkylabel)
        ax0.tick_params(labelbottom=False)

        ax1.axhline(1.0, color='black', ls='--')
        ax1.set_ylabel(r"$P_{\mathrm{pred}} / P_{\mathrm{target}}$")
        ax1.set_xlabel(self.pkxlabel)
        ax1.set_ylim(ratio_ylim)
        if k_range:
            ax0.set_xlim(k_range)

        h_col, _ = self._make_legend_handles(modes)
        ax0.legend(handles=h_col, loc='best', frameon=False)
        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, [ax0, ax1]

    def plot_comparison_maps(self, idx=0):
        """Side-by-side spatial maps of input, target and prediction."""
        reader = self.readers[0]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, mode in zip(axs, ['input', 'target', 'pred']):
            if mode not in reader.fields_dict:
                logger.warning("Mode '%s' not found in fields_dict, skipping.", mode)
                continue
            im = ax.imshow(reader.fields_dict[mode][idx], origin='lower', cmap='viridis')
            ax.set_title(self.legend_labels_dict[mode])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig, axs
