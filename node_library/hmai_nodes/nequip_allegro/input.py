import yaml
from typing import List, Dict, Tuple, Union, TypedDict

class ConfigDict(TypedDict):
    root: str
    run_name: str
    seed: int
    dataset_seed: int
    append: bool
    default_dtype: str
    model_dtype: str
    allow_tf32: bool
    model_builders: List[str]
    r_max: float
    num_layers: int
    l_max: int
    parity: bool
    num_features: int
    nonlinearity_type: str
    resnet: bool
    nonlinearity_scalars: Dict[str, str]
    nonlinearity_gates: Dict[str, str]
    num_basis: int
    BesselBasis_trainable: bool
    PolynomialCutoff_p: int
    invariant_layers: int
    invariant_neurons: int
    avg_num_neighbors: Union[str, None]
    use_sc: bool
    dataset: str
    dataset_url: str
    dataset_file_name: str
    key_mapping: Dict[str, str]
    npz_fixed_field_keys: List[str]
    chemical_symbols: List[str]
    wandb: bool
    wandb_project: str
    wandb_watch: bool
    verbose: str
    log_batch_freq: int
    log_epoch_freq: int
    save_checkpoint_freq: int
    save_ema_checkpoint_freq: int
    n_train: int
    n_val: int
    learning_rate: float
    batch_size: int
    validation_batch_size: int
    max_epochs: int
    train_val_split: str
    shuffle: bool
    metrics_key: str
    use_ema: bool
    ema_decay: float
    ema_use_num_updates: bool
    report_init_validation: bool
    early_stopping_patiences: Dict[str, int]
    early_stopping_delta: Dict[str, float]
    early_stopping_cumulative_delta: bool
    early_stopping_lower_bounds: Dict[str, float]
    early_stopping_upper_bounds: Dict[str, float]
    loss_coeffs: Dict[str, Union[float, List[Union[float, str]]]]
    metrics_components: List[List[Union[str, Dict[str, bool]]]]
    optimizer_name: str
    optimizer_amsgrad: bool
    optimizer_betas: Tuple[float, float]
    optimizer_eps: float
    optimizer_weight_decay: float
    max_gradient_norm: Union[float, None]
    lr_scheduler_name: str
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    per_species_rescale_scales_trainable: bool
    per_species_rescale_shifts_trainable: bool
    per_species_rescale_shifts: Union[str, float]
    per_species_rescale_scales: Union[None, str, float]
    global_rescale_shift: Union[None, str, float]
    global_rescale_scale: Union[None, str, float]
    global_rescale_shift_trainable: bool
    global_rescale_scale_trainable: bool

class AllegroConfig:
    """
    A class to generate a configuration file for Allegro.
    """
    
    def __init__(self):
        self.config = {
            'root': 'results/toluene',
            'run_name': 'example-run-toluene',
            'seed': 123,
            'dataset_seed': 456,
            'append': True,
            'default_dtype': 'float64',
            'model_dtype': 'float32',
            'allow_tf32': True,
            'model_builders': [
                'SimpleIrrepsConfig',
                'EnergyModel',
                'PerSpeciesRescale',
                'ForceOutput',
                'RescaleEnergyEtc'
            ],
            'r_max': 4.0,
            'num_layers': 4,
            'l_max': 1,
            'parity': True,
            'num_features': 32,
            'nonlinearity_type': 'gate',
            'resnet': False,
            'nonlinearity_scalars': {'e': 'silu', 'o': 'tanh'},
            'nonlinearity_gates': {'e': 'silu', 'o': 'tanh'},
            'num_basis': 8,
            'BesselBasis_trainable': True,
            'PolynomialCutoff_p': 6,
            'invariant_layers': 2,
            'invariant_neurons': 64,
            'avg_num_neighbors': 'auto',
            'use_sc': True,
            'dataset': 'npz',
            'dataset_url': 'http://quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip',
            'dataset_file_name': './benchmark_data/toluene_ccsd_t-train.npz',
            'key_mapping': {
                'z': 'atomic_numbers',
                'E': 'total_energy',
                'F': 'forces',
                'R': 'pos'
            },
            'npz_fixed_field_keys': ['atomic_numbers'],
            'chemical_symbols': ['H', 'C'],
            'wandb': True,
            'wandb_project': 'toluene-example',
            'wandb_watch': False,
            'verbose': 'info',
            'log_batch_freq': 100,
            'log_epoch_freq': 1,
            'save_checkpoint_freq': -1,
            'save_ema_checkpoint_freq': -1,
            'n_train': 100,
            'n_val': 50,
            'learning_rate': 0.005,
            'batch_size': 5,
            'validation_batch_size': 10,
            'max_epochs': 100000,
            'train_val_split': 'random',
            'shuffle': True,
            'metrics_key': 'validation_loss',
            'use_ema': True,
            'ema_decay': 0.99,
            'ema_use_num_updates': True,
            'report_init_validation': True,
            'early_stopping_patiences': {'validation_loss': 50},
            'early_stopping_delta': {'validation_loss': 0.005},
            'early_stopping_cumulative_delta': False,
            'early_stopping_lower_bounds': {'LR': 1.0e-5},
            'early_stopping_upper_bounds': {'cumulative_wall': 1.0e+100},
            'loss_coeffs': {
                'forces': 1.0,
                'total_energy': [1.0, 'PerAtomMSELoss']
            },
            'metrics_components': [
                [['forces', 'mae']],
                [['forces', 'rmse']],
                [['forces', 'mae'], {'PerSpecies': True, 'report_per_component': False}],
                [['forces', 'rmse'], {'PerSpecies': True, 'report_per_component': False}],
                [['total_energy', 'mae']],
                [['total_energy', 'mae'], {'PerAtom': True}]
            ],
            'optimizer_name': 'Adam',
            'optimizer_amsgrad': True,
            'optimizer_betas': (0.9, 0.999),
            'optimizer_eps': 1.0e-08,
            'optimizer_weight_decay': 0,
            'max_gradient_norm': None,
            'lr_scheduler_name': 'ReduceLROnPlateau',
            'lr_scheduler_patience': 100,
            'lr_scheduler_factor': 0.5,
            'per_species_rescale_scales_trainable': False,
            'per_species_rescale_shifts_trainable': False,
            'per_species_rescale_shifts': 'dataset_per_atom_total_energy_mean',
            'per_species_rescale_scales': None,
            'global_rescale_shift': None,
            'global_rescale_scale': 'dataset_forces_rms',
            'global_rescale_shift_trainable': False,
            'global_rescale_scale_trainable': False
        }

    @property
    def root(self) -> str:
        """Root directory of the job."""
        return self.config['root']

    @root.setter
    def root(self, value: str) -> None:
        self.config['root'] = value

    @property
    def run_name(self) -> str:
        """Run name of the job."""
        return self.config['run_name']

    @run_name.setter
    def run_name(self, value: str) -> None:
        self.config['run_name'] = value

    @property
    def seed(self) -> int:
        """Model initialization seed."""
        return self.config['seed']

    @seed.setter
    def seed(self, value: int) -> None:
        self.config['seed'] = value

    @property
    def dataset_seed(self) -> int:
        """Data set seed, determines which data to sample from file."""
        return self.config['dataset_seed']

    @dataset_seed.setter
    def dataset_seed(self, value: int) -> None:
        self.config['dataset_seed'] = value

    @property
    def append(self) -> bool:
        """Set true if a restarted run should append to the previous log file."""
        return self.config['append']

    @append.setter
    def append(self, value: bool) -> None:
        self.config['append'] = value

    @property
    def default_dtype(self) -> str:
        """Type of float to use, e.g., float32 and float64."""
        return self.config['default_dtype']

    @default_dtype.setter
    def default_dtype(self, value: str) -> None:
        self.config['default_dtype'] = value

    @property
    def model_dtype(self) -> str:
        """Data type for the model, e.g., float32."""
        return self.config['model_dtype']

    @model_dtype.setter
    def model_dtype(self, value: str) -> None:
        self.config['model_dtype'] = value

    @property
    def allow_tf32(self) -> bool:
        """Allow tf32 computation, consider setting to false if you plan to mix training/inference over any devices that are not NVIDIA Ampere or later."""
        return self.config['allow_tf32']

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        self.config['allow_tf32'] = value

    @property
    def model_builders(self) -> List[str]:
        """Tell nequip which modules to build."""
        return self.config['model_builders']

    @model_builders.setter
    def model_builders(self, value: List[str]) -> None:
        self.config['model_builders'] = value

    # Add additional properties with docstrings as needed...

    def save(self, filename: str) -> None:
        """
        Saves the configuration to a YAML file.

        Args:
            filename (str): The name of the file to save the configuration to.
        """
        with open(filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


# Usage example
config = AllegroConfig()
config.root = 'results/new_experiment'
print(config.root)  # Output: results/new_experiment
config.save('config.yaml')
