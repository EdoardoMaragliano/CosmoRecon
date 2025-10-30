import numpy as np
from glob import glob
import os
from pypower.fft_power import MeshFFTPower
from pmesh.pm import RealField, ParticleMesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

def set_latex_env():
    
    os.environ['PATH'] = r'/farmdisk1/cosmology/Libraries/texlive/2024/bin/x86_64-linux' #+ os.environ['PATH']
    # Enable LaTeX in Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Computer Modern Roman')
    rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}  % Se necessario
        \newcommand{\sfont}[1]{{\scriptscriptstyle\rm #1}}  % Definizione di \sfont
    '''

        # Set global matplotlib settings
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['axes.labelsize'] = 14
    rcParams['legend.fontsize'] = 13
    rcParams['font.size'] = 13
    rcParams['figure.figsize'] = (6.6, 4.8)

    rcParams.update({
        # pad between the legend handle (line/swatch) and the text label
        'legend.handletextpad': 0.8,      # default is 0.8

        # pad between the legend border and the content (handles + text)
        'legend.borderpad': 0.4,          # default is 0.4

        # vertical space (in fraction of font size) between legend entries
        'legend.labelspacing': 0.5,       # default is 0.5

        # (optional) pad between axes and the legend when using loc='best' etc.
        'legend.borderaxespad': 0.5,      # default is 0.5
    })


class OutputReader:
    def __init__(self, real_space=None, redshift_space=None, lt_rec=None, nn_rec=None, lt_nn_rec=None):
        """Parametri:
        real_space: path ai campi di densità in real space
        redshift_space: path ai campi di densità in redshift space
        lt_rec: path ai campi di densità ricostruiti con LT
        nn_rec: path ai campi di densità ricostruiti con NN
        lt_nn_rec: path ai campi di densità ricostruiti con LT+NN
        """

        self.path_dict = {
            'real_space': real_space,
            'redshift_space': redshift_space,
            'lt_rec': lt_rec,
            'nn_rec': nn_rec,
            'lt_nn_rec': lt_nn_rec
        }

        # Usa attributi privati
        self._fields_dict = {}
        self._pk_multipoles = {}
        self._pk_residuals = {}
        self._pk_discrepancies = {}

        print("OutputReader inizializzato")

    @property
    def fields_dict(self):
        return self._fields_dict

    @property
    def pk_multipoles(self):
        return self._pk_multipoles
    
    @pk_multipoles.setter
    def pk_multipoles(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("pk_multipoles deve essere un dizionario.")
        self._pk_multipoles = value

    @property
    def pk_residuals(self):
        return self._pk_residuals

    @pk_residuals.setter
    def pk_residuals(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("pk_residuals deve essere un dizionario.")
        self._pk_residuals = value

    @property
    def pk_discrepancies(self):
        return self._pk_discrepancies
    
    @pk_discrepancies.setter
    def pk_discrepancies(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("pk_discrepancies deve essere un dizionario.")
        self._pk_discrepancies = value

    def load_fields(self, idxs: list = None):
        """Legge i file di più tipi di simulazione (modes)"""
        for mode in self.path_dict:
            if self.path_dict[mode] is not None:
                self._load_fields(mode, idxs)
                print(f"Imported data for mode: {mode}")
            else:
                print(f"No path defined for mode: {mode}, skipping.")

    def _load_fields(self, mode: str, idxs: list = None):
        """Legge tutti i file di un tipo di simulazione (mode)"""
        if mode not in self.path_dict:
            raise ValueError(f"Mode '{mode}' non riconosciuto.")
        
        path = self.path_dict[mode]

        if path is None:
            raise ValueError(f"Nessun path definito per '{mode}'")

        all_files = sorted(glob(path))
        
        self.fields_dict[mode] = {}

        if mode in ['real_space', 'redshift_space', 'lt_rec']:
            files = [all_files[i] for i in idxs] if idxs is not None else all_files

            for i, fpath in enumerate(files):
                mock_field = np.load(fpath)
                self.fields_dict[mode][i] = mock_field
        elif mode in ['nn_rec', 'lt_nn_rec']: # they may be saved in a single file as (Nmocks, grid_size, grid_size, grid_size, 1)
            
            # check if there's only one file
            if len(all_files) == 1:
                print("Loading all NN reconstructed fields from a single file.")
                print("File path:", all_files[0])
                all_mocks = np.load(all_files[0])
                print("all_mocks shape:", all_mocks.shape, 'Formatting...')
                all_mocks = self._format_nn_rec_field(all_mocks)
                print("all_mocks formatted shape:", all_mocks.shape)
                for i in range(all_mocks.shape[0]):
                    self.fields_dict[mode][i] = all_mocks[i]

            # otherwise load them one by one
            else:
                files = [all_files[i] for i in idxs] if idxs is not None else all_files
                for i, fpath in enumerate(files):
                    mock_field = np.load(fpath)
                    self.fields_dict[mode][i] = mock_field

    def _format_nn_rec_field(self, field: np.ndarray):
        """Formatta il campo di densità ricostruito con NN"""
        # the nn output has usually shape (Nmocks, grid_size, grid_size, grid_size, 1)
        if field.ndim == 5 and field.shape[-1] == 1:
            field = np.squeeze(field, axis=-1)
        return field

    def _compute_pk_pk_multipoles(self, mock_field: np.ndarray, grid_size: int, box_size: float, box_centre: list):
        """Calcola i multipoli del PK per un singolo mock"""
        # definisci i bordi dei bin in k-space

        if np.shape(mock_field) != (grid_size, grid_size, grid_size):
            raise ValueError(f"mock_field deve avere shape {(grid_size, grid_size, grid_size)}, ma ha shape {np.shape(mock_field)}")

        kF = 2*np.pi/box_size
        kN = 2*np.pi/box_size * grid_size/2
        edges = np.arange(kF,kN, kF)

        ## define the RealFields objects
        pm = ParticleMesh(Nmesh=[grid_size, grid_size, grid_size], BoxSize=box_size)
        fieldObj = RealField(pm)

        ## assign values to the RealFields objects
        fieldObj.value[:]   = mock_field

        ## compute the power spectra
        pkObj  = MeshFFTPower(fieldObj, edges=edges, ells=(0,2,4), los='z', boxcenter=box_centre)

        k, pk_poles = pkObj.poles.k, np.array([pkObj.poles.power[0].real, pkObj.poles.power[1].real, pkObj.poles.power[2].real])

        return k, pk_poles


    def compute_all_pk_multipoles(self, modes: list, grid_size: int, box_size: float, box_centre: list):
        """Calcola i multipoli del PK tutti i mock per i modes specificati"""        

        #loop sui modes
        for mode in modes:
            print(f"Computing pk_multipoles for mode: {mode}")
            if mode not in self.fields_dict:
                raise ValueError(f"Dati per '{mode}' non caricati.")

            if mode not in self.pk_multipoles:
                self.pk_multipoles[mode] = {}

            #loop sui mock
            for i, data in tqdm(self.fields_dict[mode].items()):
                
                k, pk_poles = self._compute_pk_pk_multipoles(data, grid_size, box_size, box_centre)
                self.pk_multipoles[mode][i] = pk_poles

        self.k_values = k  # assuming k is the same for all mocks and modes. It stores the last computed k values.

    def compute_mean_pk_multipoles(self):
        """Calcola la media dei multipoli per ogni mode"""
        self.mean_pk_multipoles = {}

        for mode, multipole_dict in self.pk_multipoles.items():
            all_poles = np.array(list(multipole_dict.values()))
            mean_poles = np.mean(all_poles, axis=0)
            self.mean_pk_multipoles[mode] = mean_poles

    def compute_pk_discrepancies(self):
        """Calcola le discrepanze tra i multipoli e il real_space"""
        if 'real_space' not in self.pk_multipoles:
            raise ValueError("Multipoli real_space non calcolati.")

        # loop sui modes
        for mode in self.pk_multipoles:
            if mode == 'real_space':
                continue

            if mode not in self.pk_discrepancies:
                self.pk_discrepancies[mode] = {}

            # loop sui mock
            for i in self.pk_multipoles[mode]:
                real = self.pk_multipoles['real_space'][i] # real space pk_multipoles of mock i
                obs = self.pk_multipoles[mode][i]          # observed pk_multipoles of mock i
                self.pk_discrepancies[mode][i] = obs-real           # discrepancy for mock i


    def compute_pk_residuals(self):
        """Calcola simple stats sui residui"""
        if not self.pk_discrepancies:
            raise ValueError("Devi prima calcolare le discrepanze.")

        for mode, discr_dict in self.pk_discrepancies.items():

            discr_array = np.array(list(discr_dict.values()))
            print('discr_array shape:', discr_array.shape)
            mean_discr = np.mean(discr_array, axis=0)
            std_discr = np.std(discr_array, axis=0)

            self.pk_residuals[mode] = mean_discr/std_discr

    def compute_chi_squares(self):
        """Restituisce i chi-squared calcolati sui residui normalizzati per ogni multipolo"""

        if not hasattr(self, 'pk_residuals'):
            raise ValueError("Devi prima calcolare le statistiche dei residui.")

        self.chi_squares = {}
        for mode, res in self.pk_residuals.items():  # R shape: (3, Nk)
            self.chi_squares[mode] = {
                'mon': np.sum(res[0]**2),
                'quad': np.sum(res[1]**2),
                'hex': np.sum(res[2]**2),
            }

    
    def compute_all_stats(self, modes: list, grid_size: int, box_size: float, box_centre: list):
        """Calcola tutti i passaggi: multipoli, discrepanze, residui"""
        self.compute_all_pk_multipoles(modes, grid_size, box_size, box_centre)
        self.compute_mean_pk_multipoles()
        self.compute_pk_discrepancies()
        self.compute_pk_residuals()
        self.compute_chi_squares()
    
############################ PLOTTER CLASS #############################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Plotter:
    def __init__(self, reader, colors_dict: dict = None):
        self.reader = reader
        self.colors_dict = colors_dict if colors_dict is not None else {
            'real_space': 'black',
            'redshift_space': 'tab:orange',
            'lt_rec': 'grey',
            'nn_rec': 'tab:blue',
            'lt_nn_rec': 'tab:red'
        }

        self.pkylabel=[
            r"$P_0(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$",
            r"$P_2(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$",
            r"$P_4(k)\, [h^{-3}~\mathrm{Mpc}^{3}]$"
        ]

        self.pkxlabel = r"$k\, [h~\mathrm{Mpc}^{-1}]$"

        self.legend_labels_dict={
        'real_space': 'Real Space',
        'redshift_space': 'Redshift Space',
        'lt_rec': 'LT',
        'nn_rec': 'NN',
        'lt_nn_rec': 'LT+NN'}

    def _make_legend_handles(self, modes, legend_labels_dict=None):
        """Crea gli handles per la legenda globale."""

        handles = []
        modes.insert(0, 'real_space')  # aggiungi real_space alla legenda
        for mode in modes:
            label = legend_labels_dict.get(mode, mode) if legend_labels_dict else self.legend_labels_dict.get(mode, mode)
            color = self.colors_dict.get(mode, None)
            handle = plt.Line2D([], [], color=color, lw=2, label=label)
            handles.append(handle)

        return handles

    def plot_pk_residuals(self,
                       modes=None,
                       title=None,
                       y_label='pk_residuals',
                       figsize=(14, 4),
                       RESPANELYLIM: float = 5,
                       legend_labels_dict: dict = None,
                       share_col=True,
                       LABELSIZE: int = 11
                       ):
        """
        Plot residual stats (mean/std) for ℓ=0,2,4 pk_multipoles vs k.
        """

        if not hasattr(self.reader, 'pk_residuals'):
            raise ValueError("pk_residuals non calcolati.")

        if modes is None:
            modes = list(self.reader.pk_residuals.keys())

        share_col = 'col' if share_col else False

        k = self.reader.k_values
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=share_col)

        for l in range(3):  # ℓ = 0,2,4 pk_multipoles
            for mode in modes:
                axs[l].plot(
                    k,
                    self.reader.pk_residuals[mode][l],
                    label=mode,
                    color=self.colors_dict.get(mode, None)
                )

            axs[l].set_xlim(np.min(k), np.max(k))
            axs[l].set_xscale('log')
            axs[l].set_xlabel(self.pkxlabel)
            axs[l].set_ylabel(y_label)
            axs[l].axhline(0, linestyle="--", color="black", linewidth=1)
            axs[l].fill_between(k, -1, 1, alpha=0.2, color='grey')
            axs[l].fill_between(k, -2, 2, alpha=0.1, color='grey')
            axs[l].set_ylim(-RESPANELYLIM, RESPANELYLIM)
            axs[l].set_yticks(np.arange(-RESPANELYLIM, RESPANELYLIM+1, 2))

            axs[l].set_xscale('log')
            axs[l].set_xlabel(self.pkxlabel)
            axs[l].set_ylabel("residuals")

        # Legend unica
        handles = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(handles=handles, loc='lower center', ncol=len(modes), frameon=False, fontsize=LABELSIZE)

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig, axs

    def plot_pk_multipoles_and_residuals(self,
                                      modes=None,
                                      multipoles: list = [0, 2, 4],
                                      title=None,
                                      legend_labels_dict: dict = None,
                                      figsize=(11.7, 5),
                                      RESPANELYLIM: float = 5,
                                      UPPERPANELYLIM: list = None,
                                      width_ratios: list = None,
                                      height_ratios: list = None,  
                                      share_col: bool = True, 
                                      hspace: float = None,
                                      wspace: float = 0.2,
                                      LABELSIZE: int = 11
                                      ):
        """Plot 2xL: multipoli sopra, residui sotto."""
        
        if modes is None:
            modes = [m for m in self.reader.pk_multipoles.keys() if m != 'real_space']

        if share_col:
            share_col = 'col' 
            hspace = 0

        k = self.reader.k_values
        L = len(multipoles) #self.reader.mean_pk_multipoles[modes[0]].shape[0]
        figsize = figsize if figsize is not None else (5 * L, 8)
        

        WIDTHRATIOS = width_ratios if width_ratios is not None else [1]*L
        HEIGHTRATIOS = height_ratios if height_ratios is not None else [1]*L

        # define figure and GridSpec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, L, height_ratios=HEIGHTRATIOS, width_ratios=WIDTHRATIOS,
                               hspace=hspace, wspace=wspace)
        


        # Multipoli
        axs = np.empty(L, dtype=object)
        for l in range(L):
            
            axs[l] = fig.add_subplot(gs[0, l])

            for mode in modes + ['real_space']:
                mean_poles = self.reader.mean_pk_multipoles[mode]
                axs[l].plot(k, mean_poles[l], color=self.colors_dict.get(mode, None), linewidth=2)

            axs[l].tick_params(axis='both', which='both', direction='in', labelsize = LABELSIZE)
            axs[l].set_xscale('log')
            axs[l].set_xlim(np.min(k), np.max(k))
            axs[l].set_xlabel(self.pkxlabel) 
            axs[l].set_ylabel(self.pkylabel[l])
            if UPPERPANELYLIM is not None:
                axs[l].set_ylim(UPPERPANELYLIM) 
            if l == 0:
                axs[l].set_yscale('log')
            if l > 0:
                axs[l].axhline(0, linestyle="--", color="black", linewidth=1)
        

        # Residui
        res = np.empty(L, dtype=object)
        for l in range(L):
            if share_col == False:
                res[l] = fig.add_subplot(gs[1, l])
            elif share_col:
                res[l] = fig.add_subplot(gs[1, l], sharex=axs[l])
                axs[l].tick_params(labelbottom=False)

            for mode in modes:
                res[l].plot(k, self.reader.pk_residuals[mode][l],
                        color=self.colors_dict.get(mode, None), linewidth=2)
            res[l].set_xlim(np.min(k), np.max(k))
            res[l].set_xlabel(self.pkxlabel)

            res[l].axhline(0, linestyle="--", color="black", linewidth=1)
            res[l].fill_between(k, -1, 1, alpha=0.2, color='grey')
            res[l].fill_between(k, -2, 2, alpha=0.1, color='grey')
            res[l].set_ylim(-RESPANELYLIM, RESPANELYLIM)
            res[l].set_yticks(np.arange(-RESPANELYLIM, RESPANELYLIM+1, 2))

            res[l].set_xscale('log')
            res[l].set_xlabel(self.pkxlabel)
            res[l].set_ylabel("residuals")

        # Legend globale
        handles = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(handles=handles, loc='lower center', ncol=len(modes), bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=LABELSIZE)

        fig.suptitle(title)
        #fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()

        return fig
