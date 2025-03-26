"""Factory utily functions to create datasets and models."""

import geobreeze.foundation_models as models
from geobreeze.datasets.geobench_wrapper import GeoBenchDataset
from geobreeze.datasets.resisc_wrapper import Resics45Dataset
from geobreeze.datasets.benv2_wrapper import BenV2Dataset
from geobreeze.datasets.spectral_earth_wrapper import CorineDataset
from geobreeze.datasets.digital_typhoon_wrapper import DigitalTyphoonDataset
from geobreeze.datasets.tropical_cyclone_wrapper import TropicalCycloneDataset
from geobreeze.datasets.hyperview_wrapper import HyperviewDataset
from geobreeze.datasets.dummy_dataset import DummyWrapper
from geobreeze.engine.model import EvalModelWrapper
from geobreeze.datasets.fmow_wrapper import FmowDataset
from geobreeze.datasets.spacenet1_wrapper import SpaceNet1Dataset
model_registry = {
    "croma": models.CromaWrapper,
    "dinov2": models.DinoV2Wrapper,
    "softcon": models.SoftConWrapper,
    "dofa": models.DofaWrapper,
    # "anysat": models.AnySatWrapper,
    "senpamae": models.SenPaMAEWrapper,
    "panopticon": models.PanopticonWrapper,
    "galileo": models.GalileoWrapper,
    "anysat": models.AnySatWrapper,

    # "satmae": models.SatMAEWrapper,
    # "scalemae": models.ScaleMAEWrapper,
    # "gfm": models.GFMWrapper,
    # Add other model mappings here
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    "resisc45": Resics45Dataset,
    "benv2": BenV2Dataset,
    "corine": CorineDataset,
    "digital_typhoon": DigitalTyphoonDataset,
    "tropical_cyclone": TropicalCycloneDataset,
    "hyperview": HyperviewDataset,
    "fmow": FmowDataset,
    "dummy": DummyWrapper,
    "spacenet1": SpaceNet1Dataset,
    # Add other dataset mappings here
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")

    dataset = dataset_class(config_data)

    return dataset.create_dataset()


def create_model(model_config, dataset_config=None) -> EvalModelWrapper:
    model_name = model_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    model = model_class(model_config, dataset_config)

    return model
