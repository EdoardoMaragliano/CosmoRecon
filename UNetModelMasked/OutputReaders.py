import numpy as np
from glob import glob
import os
from pypower.fft_power import MeshFFTPower
from pmesh.pm import RealField, ParticleMesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        self._data_dict = {}
        self._multipoles = {}
        self._residuals = {}
        self._discrepancies = {}

        print("OutputReader inizializzato")

    @property
    def data_dict(self):
        return self._data_dict

    @property
    def multipoles(self):
        return self._multipoles
    
    @multipoles.setter
    def multipoles(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("multipoles deve essere un dizionario.")
        self._multipoles = value

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("residuals deve essere un dizionario.")
        self._residuals = value

    @property
    def discrepancies(self):
        return self._discrepancies
    
    @discrepancies.setter
    def discrepancies(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("discrepancies deve essere un dizionario.")
        self._discrepancies = value

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
        
        self.data_dict[mode] = {}

        if mode in ['real_space', 'redshift_space', 'lt_rec']:
            files = [all_files[i] for i in idxs] if idxs is not None else all_files

            for i, fpath in enumerate(files):
                mock_field = np.load(fpath)
                self.data_dict[mode][i] = mock_field
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
                    self.data_dict[mode][i] = all_mocks[i]

            # otherwise load them one by one
            else:
                files = [all_files[i] for i in idxs] if idxs is not None else all_files
                for i, fpath in enumerate(files):
                    mock_field = np.load(fpath)
                    self.data_dict[mode][i] = mock_field

    def _format_nn_rec_field(self, field: np.ndarray):
        """Formatta il campo di densità ricostruito con NN"""
        # the nn output has usually shape (Nmocks, grid_size, grid_size, grid_size, 1)
        if field.ndim == 5 and field.shape[-1] == 1:
            field = np.squeeze(field, axis=-1)
        return field

    def _compute_pk_multipoles(self, mock_field: np.ndarray, grid_size: int, box_size: float, box_centre: list):
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


    def compute_all_multipoles(self, modes: list, grid_size: int, box_size: float, box_centre: list):
        """Calcola i multipoli del PK tutti i mock per i modes specificati"""        

        #loop sui modes
        for mode in modes:
            print(f"Computing multipoles for mode: {mode}")
            if mode not in self.data_dict:
                raise ValueError(f"Dati per '{mode}' non caricati.")

            if mode not in self.multipoles:
                self.multipoles[mode] = {}

            #loop sui mock
            for i, data in tqdm(self.data_dict[mode].items()):
                
                k, pk_poles = self._compute_pk_multipoles(data, grid_size, box_size, box_centre)
                self.multipoles[mode][i] = pk_poles

        self.k_values = k  # assuming k is the same for all mocks and modes. It stores the last computed k values.

    def compute_mean_multipoles(self):
        """Calcola la media dei multipoli per ogni mode"""
        self.mean_multipoles = {}

        for mode, multipole_dict in self.multipoles.items():
            all_poles = np.array(list(multipole_dict.values()))
            mean_poles = np.mean(all_poles, axis=0)
            self.mean_multipoles[mode] = mean_poles

    def compute_discrepancies(self):
        """Calcola le discrepanze tra i multipoli e il real_space"""
        if 'real_space' not in self.multipoles:
            raise ValueError("Multipoli real_space non calcolati.")

        # loop sui modes
        for mode in self.multipoles:
            if mode == 'real_space':
                continue

            if mode not in self.discrepancies:
                self.discrepancies[mode] = {}

            # loop sui mock
            for i in self.multipoles[mode]:
                real = self.multipoles['real_space'][i] # real space multipoles of mock i
                obs = self.multipoles[mode][i]          # observed multipoles of mock i
                self.discrepancies[mode][i] = obs-real           # discrepancy for mock i


    def compute_residuals(self):
        """Calcola simple stats sui residui"""
        if not self.discrepancies:
            raise ValueError("Devi prima calcolare le discrepanze.")

        for mode, discr_dict in self.discrepancies.items():

            discr_array = np.array(list(discr_dict.values()))
            print('discr_array shape:', discr_array.shape)
            mean_discr = np.mean(discr_array, axis=0)
            std_discr = np.std(discr_array, axis=0)

            self.residuals[mode] = mean_discr/std_discr
        
   

    def get_chi_squares(self):
        """Restituisce i chi-squared calcolati sui residui normalizzati per ogni multipolo"""

        if not hasattr(self, 'residuals'):
            raise ValueError("Devi prima calcolare le statistiche dei residui.")

        self.chi_squares = {}
        for mode, res in self.residuals.items():  # R shape: (3, Nk)
            self.chi_squares[mode] = {
                'mon': np.sum(res[0]**2),
                'quad': np.sum(res[1]**2),
                'hex': np.sum(res[2]**2),
            }

        return self.chi_squares

    


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Plotter:
    def __init__(self, reader, colors_dict: dict = None):
        self.reader = reader
        self.colors_dict = colors_dict if colors_dict is not None else {
            'real_space': 'black',
            'redshift_space': 'green',
            'lt_rec': 'grey',
            'nn_rec': 'blue',
            'lt_nn_rec': 'red'
        }

    def _make_legend_handles(self, modes, legend_labels_dict=None):
        """Crea gli handles per la legenda globale."""

        handles = []
        modes.insert(0, 'real_space')  # aggiungi real_space alla legenda
        for mode in modes:
            label = legend_labels_dict.get(mode, mode) if legend_labels_dict else mode
            color = self.colors_dict.get(mode, None)
            handle = plt.Line2D([], [], color=color, lw=2, label=label)
            handles.append(handle)

        return handles

    def plot_residuals(self,
                       modes=None,
                       title='Residuals Statistics across all mocks',
                       y_label='Residuals (Mean/Std)',
                       figsize=(14, 4),
                       legend_labels_dict: dict = None):
        """
        Plot residual stats (mean/std) for ℓ=0,2,4 multipoles vs k.
        """

        if not hasattr(self.reader, 'residuals'):
            raise ValueError("Residuals non calcolati.")

        if modes is None:
            modes = list(self.reader.residuals.keys())

        k = self.reader.k_values
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        for l in range(3):  # ℓ = 0,2,4 multipoles
            for mode in modes:
                axs[l].plot(
                    k,
                    self.reader.residuals[mode][l],
                    label=mode,
                    color=self.colors_dict.get(mode, None)
                )

            axs[l].set_xscale('log')
            axs[l].set_xlabel('$k \mathrm{[hMpc^{-1}]}$')
            axs[l].set_ylabel(y_label)
            axs[l].set_title(f'ℓ={2*l}')
            axs[l].axhline(0, color='black', linestyle='--')
            axs[l].fill_between(k, -1, 1, alpha=0.15)
            axs[l].fill_between(k, -2, 2, alpha=0.10)

        # Legend unica
        handles = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(handles=handles, loc='lower center', ncol=len(modes))

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig, axs

    def plot_multipoles_and_residuals(self,
                                      modes=None,
                                      legend_labels_dict: dict = None,
                                      figsize=(15, 10),
                                      RESYLIM=5,
                                      multipoles: list = [0, 2, 4]):
        """Plot 2xL: multipoli sopra, residui sotto."""
        
        if modes is None:
            modes = [m for m in self.reader.multipoles.keys() if m != 'real_space']

        k = self.reader.k_values
        L = self.reader.mean_multipoles[modes[0]].shape[0]
        figsize = figsize if figsize is not None else (5 * L, 8)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, L, height_ratios=[3, 1])

        # Multipoli
        for l in range(L):
            ax = fig.add_subplot(gs[0, l])
            for mode in modes + ['real_space']:
                mean_poles = self.reader.mean_multipoles[mode]
                ax.plot(k, mean_poles[l], color=self.colors_dict.get(mode, None))

            ax.set_xscale('log')
            ax.set_xlim(np.min(k), np.max(k))
            ax.set_xlabel("$k\, [h\, \mathrm{Mpc}^{-1}]$")
            ylabel = "$P_{" + str(2*l) + "}(k)\, [h^3Mpc^{-3}]$"
            ax.set_ylabel(ylabel)
            if l == 0:
                ax.set_yscale('log')
            if l > 0:
                ax.axhline(0, linestyle="--", color="black", linewidth=1)
        

        # Residui
        for l in range(L):
            ax = fig.add_subplot(gs[1, l])
            for mode in modes:
                ax.plot(k, self.reader.residuals[mode][l],
                        color=self.colors_dict.get(mode, None))
            ax.set_xlim(np.min(k), np.max(k))
            ax.set_xlabel("$k\, [h\, \mathrm{Mpc}^{-1}]$")

            ax.axhline(0, linestyle="--", color="black", linewidth=1)
            ax.fill_between(k, -1, 1, alpha=0.2, color='grey')
            ax.fill_between(k, -2, 2, alpha=0.1, color='grey')
            ax.set_ylim(-RESYLIM, RESYLIM)

            ax.set_xscale('log')
            ax.set_xlabel("$k\, [h\, \mathrm{Mpc}^{-1}]$ ")
            ax.set_ylabel("Residuals")

        # Legend globale
        handles = self._make_legend_handles(modes, legend_labels_dict)
        fig.legend(handles=handles, loc='lower center', ncol=len(modes))

        fig.suptitle("Multipoles (top) & Normalized Residuals (bottom)")
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()

        return fig
