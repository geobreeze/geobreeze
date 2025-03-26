import torch
from typing import Union, NamedTuple, List, OrderedDict as OrderedDictType
from typing import Optional, cast
from torch import Tensor
from torch import nn

from collections import OrderedDict
import math
import numpy as np
from copy import deepcopy


DEFAULT_MONTH = 5


# band information
S1_BANDS = ["VV", "VH"]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
TC_BANDS = ["def", "soil", "aet"]
VIIRS_BANDS = ["avg_rad"]
SRTM_BANDS = ["elevation", "slope"]
DW_BANDS = [
    "DW_water",
    "DW_trees",
    "DW_grass",
    "DW_flooded_vegetation",
    "DW_crops",
    "DW_shrub_and_scrub",
    "DW_built",
    "DW_bare",
    "DW_snow_and_ice",
]
WC_BANDS = [
    "WC_temporarycrops",
    "WC_maize",
    "WC_wintercereals",
    "WC_springcereals",
    "WC_irrigation",
]
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]

LANDSCAN_BANDS = ["b1"]
LOCATION_BANDS = ["x", "y", "z"]

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS


SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


def to_cartesian(
    lat: Union[float, np.ndarray, torch.Tensor],
    lon: Union[float, np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(lat, float):
        assert -90 <= lat <= 90, (
            f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon <= 180, (
            f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), (
            f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        )
        assert 90 >= lat.max(), (
            f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon.min(), (
            f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        )
        assert 180 >= lon.max(), (
            f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert -90 <= lat.min(), (
            f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        )
        assert 90 >= lat.max(), (
            f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon.min(), (
            f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        )
        assert 180 >= lon.max(), (
            f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")


class Normalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    std_bands: dict[int, list] = {
        len(SPACE_TIME_BANDS): [b for b in SPACE_TIME_BANDS if b != "NDVI"],
        len(SPACE_BANDS): SRTM_BANDS,
        len(TIME_BANDS): TIME_BANDS,
        len(STATIC_BANDS): LANDSCAN_BANDS,
    }

    def __init__(
        self,
        std: bool = True,
        normalizing_dicts: Optional[dict] = None,
        std_multiplier: float = 2,
    ):
        self.shift_div_dict = {
            len(SPACE_TIME_BANDS): {
                "shift": deepcopy(SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(SPACE_TIME_DIV_VALUES),
            },
            len(SPACE_BANDS): {
                "shift": deepcopy(SPACE_SHIFT_VALUES),
                "div": deepcopy(SPACE_DIV_VALUES),
            },
            len(TIME_BANDS): {
                "shift": deepcopy(TIME_SHIFT_VALUES),
                "div": deepcopy(TIME_DIV_VALUES),
            },
            len(STATIC_BANDS): {
                "shift": deepcopy(STATIC_SHIFT_VALUES),
                "div": deepcopy(STATIC_DIV_VALUES),
            },
        }
        print(self.shift_div_dict.keys())
        self.normalizing_dicts = normalizing_dicts
        if std:
            name_to_bands = {
                len(SPACE_TIME_BANDS): SPACE_TIME_BANDS,
                len(SPACE_BANDS): SPACE_BANDS,
                len(TIME_BANDS): TIME_BANDS,
                len(STATIC_BANDS): STATIC_BANDS,
            }
            assert normalizing_dicts is not None
            for key, val in normalizing_dicts.items():
                if isinstance(key, str):
                    continue
                bands_to_replace = self.std_bands[key]
                for band in bands_to_replace:
                    band_idx = name_to_bands[key].index(band)
                    mean = val["mean"][band_idx]
                    std = val["std"][band_idx]
                    min_value = mean - (std_multiplier * std)
                    max_value = mean + (std_multiplier * std)
                    div = max_value - min_value
                    if div == 0:
                        raise ValueError(f"{band} has div value of 0")
                    self.shift_div_dict[key]["shift"][band_idx] = min_value
                    self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(
        x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray
    ) -> np.ndarray:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: np.ndarray):
        div_values = self.shift_div_dict[x.shape[-1]]["div"]
        return self._normalize(x, self.shift_div_dict[x.shape[-1]]["shift"], div_values)


class MaskedOutput(NamedTuple):
    """
    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """

    space_time_x: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS)]
    space_x: torch.Tensor  # [B, H, W, len(SPACE_BANDS)]
    time_x: torch.Tensor  # [B, T, len(TIME_BANDS)]
    static_x: torch.Tensor  # [B, len(STATIC_BANDS)]
    space_time_mask: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX)]
    space_mask: torch.Tensor  # [B, H, W, len(SPACE_BAND_GROUPS_IDX)]
    time_mask: torch.Tensor  # [B, T, len(TIME_BAND_GROUPS_IDX)]
    static_mask: torch.Tensor  # [B, len(STATIC_BAND_GROUPS_IDX)]
    months: torch.Tensor  # [B, T]


def construct_galileo_input(
    s1: torch.Tensor | None = None,  # [H, W, T, D]
    s2: torch.Tensor | None = None,  # [H, W, T, D]
    era5: torch.Tensor | None = None,  # [T, D]
    tc: torch.Tensor | None = None,  # [T, D]
    viirs: torch.Tensor | None = None,  # [T, D]
    srtm: torch.Tensor | None = None,  # [H, W, D]
    dw: torch.Tensor | None = None,  # [H, W, D]
    wc: torch.Tensor | None = None,  # [H, W, D]
    landscan: torch.Tensor | None = None,  # [D]
    latlon: torch.Tensor | None = None,  # [D]
    months: torch.Tensor | None = None,  # [T]
    normalize: bool = False,
):
    space_time_inputs = [s1, s2]
    time_inputs = [era5, tc, viirs]
    space_inputs = [srtm, dw, wc]
    static_inputs = [landscan, latlon]
    devices = [
        x.device
        for x in space_time_inputs + time_inputs + space_inputs + static_inputs
        if x is not None
    ]

    if len(devices) == 0:
        raise ValueError("At least one input must be not None")
    if not all(devices[0] == device for device in devices):
        raise ValueError("Received tensors on multiple devices")
    device = devices[0]

    # first, check all the input shapes are consistent
    timesteps_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in time_inputs if x is not None
    ]
    height_list = [x.shape[0] for x in space_time_inputs if x is not None] + [
        x.shape[0] for x in space_inputs if x is not None
    ]
    width_list = [x.shape[1] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in space_inputs if x is not None
    ]

    if len(timesteps_list) > 0:
        if not all(timesteps_list[0] == timestep for timestep in timesteps_list):
            raise ValueError("Inconsistent number of timesteps per input")
        t = timesteps_list[0]
    else:
        t = 1

    if len(height_list) > 0:
        if not all(height_list[0] == height for height in height_list):
            raise ValueError("Inconsistent heights per input")
        if not all(width_list[0] == width for width in width_list):
            raise ValueError("Inconsistent widths per input")
        h = height_list[0]
        w = width_list[0]
    else:
        h, w = 1, 1

    # now, we can construct our empty input tensors. By default, everything is masked
    s_t_x = torch.zeros(
        (h, w, t, len(SPACE_TIME_BANDS)), dtype=torch.float, device=device
    )
    s_t_m = torch.ones(
        (h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=torch.float, device=device
    )
    sp_x = torch.zeros((h, w, len(SPACE_BANDS)), dtype=torch.float, device=device)
    sp_m = torch.ones(
        (h, w, len(SPACE_BAND_GROUPS_IDX)), dtype=torch.float, device=device
    )
    t_x = torch.zeros((t, len(TIME_BANDS)), dtype=torch.float, device=device)
    t_m = torch.ones((t, len(TIME_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    st_x = torch.zeros((len(STATIC_BANDS)), dtype=torch.float, device=device)
    st_m = torch.ones((len(STATIC_BAND_GROUPS_IDX)), dtype=torch.float, device=device)

    for x, bands_list, group_key in zip([s1, s2], [S1_BANDS, S2_BANDS], ["S1", "S2"]):
        if x is not None:
            indices = [
                idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in bands_list
            ]
            groups_idx = [
                idx
                for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX)
                if group_key in key
            ]
            s_t_x[:, :, :, indices] = x
            s_t_m[:, :, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [srtm, dw, wc], [SRTM_BANDS, DW_BANDS, WC_BANDS], ["SRTM", "DW", "WC"]
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(SPACE_BAND_GROUPS_IDX) if group_key in key
            ]
            sp_x[:, :, indices] = x
            sp_m[:, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [era5, tc, viirs], [ERA5_BANDS, TC_BANDS, VIIRS_BANDS], ["ERA5", "TC", "VIIRS"]
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(TIME_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(TIME_BAND_GROUPS_IDX) if group_key in key
            ]
            t_x[:, indices] = x
            t_m[:, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [landscan, latlon], [LANDSCAN_BANDS, LOCATION_BANDS], ["LS", "location"]
    ):
        if x is not None:
            if group_key == "location":
                # transform latlon to cartesian
                x = cast(torch.Tensor, to_cartesian(x[0], x[1]))
            indices = [idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list]
            groups_idx = [
                idx
                for idx, key in enumerate(STATIC_BAND_GROUPS_IDX)
                if group_key in key
            ]
            st_x[indices] = x
            st_m[groups_idx] = 0

    if months is None:
        months = torch.ones((t), dtype=torch.long, device=device) * DEFAULT_MONTH
    else:
        if months.shape[0] != t:
            raise ValueError("Incorrect number of input months")

    if normalize:
        normalizer = Normalizer(std=False)
        s_t_x = torch.from_numpy(normalizer(s_t_x.cpu().numpy())).to(device)
        sp_x = torch.from_numpy(normalizer(sp_x.cpu().numpy())).to(device)
        t_x = torch.from_numpy(normalizer(t_x.cpu().numpy())).to(device)
        st_x = torch.from_numpy(normalizer(st_x.cpu().numpy())).to(device)

    return MaskedOutput(
        space_time_x=s_t_x,
        space_time_mask=s_t_m,
        space_x=sp_x,
        space_mask=sp_m,
        time_x=t_x,
        time_mask=t_m,
        static_x=st_x,
        static_mask=st_m,
        months=months,
    )


def to_cartesian(
    lat: Union[float, np.ndarray, torch.Tensor],
    lon: Union[float, np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(lat, float):
        assert -90 <= lat <= 90, (
            f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon <= 180, (
            f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), (
            f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        )
        assert 90 >= lat.max(), (
            f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon.min(), (
            f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        )
        assert 180 >= lon.max(), (
            f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert -90 <= lat.min(), (
            f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        )
        assert 90 >= lat.max(), (
            f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon.min(), (
            f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        )
        assert 180 >= lon.max(), (
            f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        )
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")
