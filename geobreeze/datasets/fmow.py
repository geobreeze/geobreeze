from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.samplers.utils import _to_tuple
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
from PIL import Image
import logging
from tifffile import imread
import kornia.augmentation as K
import traceback
from .base import BaseDataset

logger = logging.getLogger()

class FmowBenchmarkDataset(NonGeoDataset):

    # Define class-level constants for mean and standard deviation of each band
    MEAN: dict = { "fmow_s2" : torch.tensor([1569.1970, 1383.9951, 1338.4231, 1408.1113, 1537.2856,
                                 1926.5183, 2136.5127, 2033.4019, 2262.4558, 674.1297, 
                                 16.7465, 2027.3674, 1501.4686]) ,
            "fmow_wv23": torch.tensor([324.14698805992043, 321.9731490132023, 414.5839148154745, 376.7135672751123, 287.4754539285566, 400.37182819120585, 455.80341042344526, 387.41375747632117]), #over 20% of train, 55152 samples 
            "fmow_qbge": torch.tensor([444.3088455498445, 427.3245457864162, 306.4215948690728, 576.8987799591143]), #over 40% of train, 35123 samples 
            "fmow_rgb":  torch.tensor([104.55944194258717, 106.19051077034885, 100.09524832336331]), # rgb highres over 10% of train (all 4 sensors, 36357 samples)
    }

    STD: dict = { "fmow_s2" : torch.tensor([517.7932, 595.0351, 690.2477, 994.7734, 994.4599, 1079.4558, 1192.3668,
                    1170.9979, 1278.2103, 474.7933, 15.1294, 1362.3807, 1134.5983]),
            "fmow_wv23": torch.tensor([115.78370553994293, 134.64109966204413, 210.5490582067263, 242.05723188930372, 204.35781294894136, 246.31516378243006, 282.0780383349591, 255.94032664657144]),
            "fmow_qbge": torch.tensor([229.0287, 247.0869, 239.2377, 381.1768]),   
            "fmow_rgb":  torch.tensor([68.97789839437421, 66.91165970478865, 69.09505694641828]), # rgb highres over 10% of train (all 4 sensors, 36357 samples)
    }

    sensor_name_mapping: dict = {'WORLDVIEW02' : 'fmow_wv23', 
                                 'WORLDVIEW03_VNIR' : 'fmow_wv23',
                                 'GEOEYE01': 'fmow_qbge',
                                 'QUICKBIRD02': 'fmow_qbge',
                                 'fmow_rgb': 'fmow_rgb'
                                }
    RGB_CHANNELS: dict = {
        'fmow_s2': [3, 2, 1],
        'fmow_wv23': [4, 2, 1], 
        'fmow_rgb': [0, 1, 2],  
    }

    CLASS_NAMES = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
    #It was found empirically that some wv2 or wv3 images are not 8 channels, so for simplicty we just drop them
    PROBLEMATIC_IDS = { 'train' :
                        {8: ['shipyard_10_7', 'water_treatment_facility_1468_0', 'water_treatment_facility_208_2', 'water_treatment_facility_1315_0', 'airport_65_0', 'airport_345_0', 'park_170_0', 'park_461_1', 'park_276_0', 'park_635_3', 'park_610_1', 'park_619_0', 'park_699_1', 'park_645_0', 'park_710_0', 'park_390_0', 'solar_farm_284_1', 'solar_farm_309_0', 'solar_farm_850_3', 'solar_farm_1678_0', 'tower_349_3', 'parking_lot_or_garage_741_4', 'parking_lot_or_garage_2025_0', 'parking_lot_or_garage_775_3', 'parking_lot_or_garage_1744_0', 'parking_lot_or_garage_667_5', 'parking_lot_or_garage_277_0', 'parking_lot_or_garage_836_4', 'parking_lot_or_garage_985_4', 'parking_lot_or_garage_129_0', 'parking_lot_or_garage_117_2', 'parking_lot_or_garage_1176_0', 'parking_lot_or_garage_1840_0', 'parking_lot_or_garage_992_0', 'parking_lot_or_garage_891_0', 'parking_lot_or_garage_398_4', 'parking_lot_or_garage_1958_0', 'parking_lot_or_garage_1356_0', 'parking_lot_or_garage_1164_0', 'parking_lot_or_garage_993_0', 'parking_lot_or_garage_407_0', 'parking_lot_or_garage_1104_0', 'parking_lot_or_garage_325_4', 'parking_lot_or_garage_101_4', 'parking_lot_or_garage_2606_6', 'parking_lot_or_garage_1339_4', 'parking_lot_or_garage_435_4', 'parking_lot_or_garage_629_2', 'multi-unit_residential_625_1', 'multi-unit_residential_625_0', 'multi-unit_residential_559_1', 'road_bridge_718_0', 'road_bridge_89_0', 'road_bridge_580_1', 'road_bridge_464_0', 'helipad_215_4', 'helipad_930_0', 'helipad_675_0', 'helipad_696_0', 'helipad_300_4', 'helipad_96_5', 'runway_370_1', 'runway_214_0', 'crop_field_1500_1', 'crop_field_1766_2', 'crop_field_271_3', 'crop_field_288_3', 'crop_field_1730_3', 'crop_field_1255_4', 'crop_field_2765_3', 'crop_field_1717_3', 'airport_terminal_971_0', 'airport_terminal_904_0', 'airport_terminal_751_0', 'airport_terminal_318_0', 'swimming_pool_893_4', 'ground_transportation_station_1244_0', 'ground_transportation_station_1083_12', 'ground_transportation_station_67_13', 'ground_transportation_station_259_16', 'ground_transportation_station_1514_0', 'ground_transportation_station_185_1', 'ground_transportation_station_474_3', 'ground_transportation_station_1330_0', 'ground_transportation_station_996_12', 'ground_transportation_station_776_7', 'ground_transportation_station_1212_20', 'interchange_134_18', 'interchange_322_0', 'interchange_346_4', 'interchange_173_0', 'interchange_399_0', 'interchange_277_13', 'interchange_220_4', 'police_station_967_0', 'police_station_721_1', 'police_station_1010_0', 'zoo_27_0', 'zoo_244_0', 'zoo_161_0', 'recreational_facility_3292_5', 'recreational_facility_659_4', 'recreational_facility_871_3', 'recreational_facility_1377_0', 'recreational_facility_4607_0', 'recreational_facility_2455_2', 'recreational_facility_2134_0', 'recreational_facility_4175_0', 'recreational_facility_4100_1', 'recreational_facility_1569_2', 'recreational_facility_4466_0', 'recreational_facility_542_0', 'recreational_facility_4294_0', 'recreational_facility_2046_3', 'recreational_facility_3827_0', 'recreational_facility_3332_0', 'recreational_facility_4693_0', 'recreational_facility_2824_0', 'recreational_facility_1358_0', 'recreational_facility_4709_0', 'recreational_facility_1087_0', 'recreational_facility_3285_4', 'recreational_facility_531_3', 'recreational_facility_2120_0', 'recreational_facility_3764_0', 'recreational_facility_979_0', 'recreational_facility_3982_0', 'recreational_facility_4275_0', 'recreational_facility_2308_4', 'recreational_facility_2313_0', 'recreational_facility_2538_8', 'recreational_facility_2375_0', 'recreational_facility_3967_1', 'recreational_facility_1870_0', 'recreational_facility_194_0', 'recreational_facility_1291_0', 'recreational_facility_1411_0', 'recreational_facility_3319_0', 'recreational_facility_3815_0', 'recreational_facility_1670_0', 'recreational_facility_1299_0', 'recreational_facility_3230_0', 'recreational_facility_4552_0', 'recreational_facility_2966_0', 'recreational_facility_579_0', 'recreational_facility_4237_0', 'recreational_facility_3846_0', 'recreational_facility_1338_4', 'recreational_facility_261_2', 'recreational_facility_833_4', 'recreational_facility_1684_0', 'recreational_facility_3204_0', 'recreational_facility_4134_0', 'prison_100_2', 'prison_107_0', 'prison_13_0', 'prison_260_0', 'oil_or_gas_facility_883_0', 'oil_or_gas_facility_1445_0', 'railway_bridge_689_0', 'railway_bridge_681_0', 'railway_bridge_187_0', 'railway_bridge_140_0', 'railway_bridge_99_0', 'railway_bridge_509_12', 'railway_bridge_210_0', 'railway_bridge_783_0', 'smokestack_586_0', 'smokestack_140_5', 'smokestack_55_0', 'educational_institution_1210_0', 'educational_institution_1337_0', 'educational_institution_1837_0', 'educational_institution_1518_0', 'educational_institution_288_0', 'educational_institution_615_2', 'educational_institution_198_4', 'educational_institution_1708_0', 'educational_institution_294_5', 'educational_institution_315_0', 'educational_institution_1811_0', 'educational_institution_967_12', 'educational_institution_740_5', 'educational_institution_480_13', 'educational_institution_1769_1', 'educational_institution_368_1', 'educational_institution_1217_4', 'educational_institution_1449_0', 'educational_institution_1153_4', 'educational_institution_1350_1', 'educational_institution_1568_1', 'place_of_worship_3515_17', 'place_of_worship_2117_0', 'place_of_worship_3562_7', 'place_of_worship_2023_0', 'place_of_worship_837_0', 'place_of_worship_4644_0', 'place_of_worship_1513_0', 'place_of_worship_3532_0', 'place_of_worship_1432_2', 'lighthouse_101_4', 'stadium_634_0', 'stadium_503_0', 'stadium_555_0', 'stadium_112_0', 'stadium_274_0', 'stadium_355_4', 'stadium_518_4', 'stadium_150_0', 'stadium_452_4', 'flooded_road_421_3', 'gas_station_922_0', 'gas_station_960_0', 'gas_station_356_0', 'gas_station_798_0', 'gas_station_447_0', 'gas_station_1043_0', 'military_facility_390_2', 'military_facility_1590_0', 'military_facility_776_0', 'military_facility_1147_0', 'military_facility_995_0', 'military_facility_1395_0', 'military_facility_460_0', 'military_facility_502_0', 'military_facility_1124_0', 'debris_or_rubble_259_0', 'storage_tank_550_7', 'storage_tank_464_0', 'shopping_mall_346_4', 'shopping_mall_434_3', 'shopping_mall_158_0', 'shopping_mall_843_0', 'shopping_mall_129_4', 'electric_substation_943_2', 'electric_substation_800_0', 'electric_substation_960_0', 'electric_substation_183_1', 'electric_substation_244_3', 'airport_hangar_299_4', 'airport_hangar_806_0', 'airport_hangar_22_0', 'amusement_park_1057_0', 'amusement_park_1127_0', 'amusement_park_294_5', 'amusement_park_399_0', 'burial_site_322_0', 'burial_site_998_0', 'burial_site_252_4', 'burial_site_475_5', 'burial_site_937_0', 'burial_site_26_5', 'burial_site_74_5', 'waste_disposal_598_0', 'waste_disposal_805_0', 'waste_disposal_848_0', 'toll_booth_602_0', 'toll_booth_1133_0', 'toll_booth_1027_0', 'construction_site_478_0', 'golf_course_70_1', 'golf_course_75_5', 'golf_course_268_0', 'golf_course_7_5', 'golf_course_638_0', 'golf_course_28_13', 'golf_course_554_0', 'golf_course_265_0', 'golf_course_365_2', 'golf_course_65_0', 'golf_course_207_2', 'golf_course_487_0', 'golf_course_148_2', 'golf_course_52_1', 'hospital_632_0', 'hospital_49_0', 'hospital_429_0', 'hospital_84_0', 'hospital_982_0', 'hospital_426_0', 'hospital_303_0', 'hospital_209_16', 'car_dealership_1222_0', 'car_dealership_242_1', 'car_dealership_729_0', 'car_dealership_297_2', 'car_dealership_1179_0', 'office_building_547_0', 'office_building_676_0', 'office_building_961_0', 'office_building_108_0', 'office_building_157_0', 'surface_mine_711_0', 'surface_mine_1107_0', 'surface_mine_1450_0', 'race_track_1028_0', 'race_track_523_0', 'race_track_72_0', 'race_track_650_0', 'race_track_943_0', 'race_track_581_0', 'race_track_524_0', 'race_track_789_0']},
                        'val': 
                        {8: ['park_102_1', 'park_8_3', 'solar_farm_96_1', 'parking_lot_or_garage_139_1', 'parking_lot_or_garage_203_0', 'parking_lot_or_garage_43_2', 'parking_lot_or_garage_129_0', 'parking_lot_or_garage_160_0', 'parking_lot_or_garage_97_0', 'parking_lot_or_garage_191_0', 'parking_lot_or_garage_159_0', 'helipad_126_0', 'crop_field_395_0', 'crop_field_38_0', 'airport_terminal_124_1', 'interchange_65_0', 'police_station_111_0', 'recreational_facility_589_0', 'recreational_facility_271_0', 'recreational_facility_92_0', 'recreational_facility_384_0', 'recreational_facility_599_0', 'recreational_facility_231_3', 'recreational_facility_360_1', 'recreational_facility_466_4', 'recreational_facility_0_0', 'recreational_facility_116_0', 'recreational_facility_269_0', 'recreational_facility_614_0', 'railway_bridge_16_14', 'smokestack_11_0', 'educational_institution_100_0', 'educational_institution_207_0', 'educational_institution_221_0', 'stadium_0_19', 'stadium_0_18', 'military_facility_45_0', 'fountain_52_0', 'electric_substation_73_0', 'electric_substation_100_1', 'electric_substation_128_5', 'electric_substation_105_5', 'electric_substation_116_0', 'waste_disposal_60_0', 'toll_booth_115_0', 'golf_course_48_0', 'hospital_68_0', 'car_dealership_166_0', 'car_dealership_112_1']},
                        'test': 
                        {8: ['water_treatment_facility_2_0', 'park_53_0', 'tower_179_0', 'parking_lot_or_garage_72_0', 'parking_lot_or_garage_14_3', 'parking_lot_or_garage_226_1', 'road_bridge_86_0', 'helipad_84_0', 'border_checkpoint_9_6', 'airport_terminal_75_0', 'ground_transportation_station_212_15', 'ground_transportation_station_21_1', 'interchange_67_0', 'recreational_facility_314_0', 'recreational_facility_325_2', 'recreational_facility_293_0', 'prison_54_4', 'educational_institution_189_1', 'educational_institution_39_1', 'educational_institution_248_0', 'place_of_worship_53_0', 'gas_station_156_0', 'electric_substation_157_1', 'waste_disposal_6_0', 'waste_disposal_129_0', 'waste_disposal_88_0', 'toll_booth_129_0', 'golf_course_8_0', 'golf_course_76_3', 'hospital_102_2', 'office_building_138_0', 'office_building_36_0', 'office_building_29_0', 'race_track_121_0']},
                        
                        }
    
    def __init__(self, split='train', 
                 normalize=True, 
                 root='${RDIR}/datasets',
                 keep_sensors=['WORLDVIEW02', 'WORLDVIEW03_VNIR', 'GEOEYE01', 'QUICKBIRD02'],
                 transforms=None,
                 full_spectra=False,
                 output_dtype='float32',
                 min_img_size:int = None,
                 max_img_size:int = None,
                 num_channels:int = 4):
        """

        split: str, one of ['train', 'val', 'test']
        normalize: bool, whether to normalize the images
        root: str, root directory of the dataset
        transform: callable, transform to apply to the images
        max_crop: int, each row must have its minimum size >= this (prevents fmow from generating all images with a small size)
        min_crop: int, each row must have all its image size >= this (ensures fmow generating all images above this min threshold)
        keep_sensors: list of str, sensors to keep in the dataset, one of ['wv23', 's2', 'rgb']
        """

        root = os.path.expandvars(root)
        self.transforms = transforms
        self.return_rgb = False
        self.num_classes = len(self.CLASS_NAMES)
        self.num_channels = num_channels

        assert isinstance(output_dtype, str)
        if output_dtype == 'float16':
            torch_dtype, np_dtype = torch.float16, np.float16
        elif output_dtype == 'float32':
            torch_dtype, np_dtype = torch.float32, np.float32
        elif output_dtype == 'float64':
            torch_dtype, np_dtype = torch.float64, np.float64
        else:
            raise ValueError(f'Unknown output_dtype: {output_dtype}')
        self.torch_dtype, self.np_dtype = torch_dtype, np_dtype
        logger.info(f'output_dtype: {output_dtype}')

        # read file metainfo
        if split in ['train', 'val']: 
            metadata_path = os.path.join(root, f'fmow/metadata_v2/fmow_{split}.parquet')
        elif split is None: #pick train
            metadata_path = os.path.join(root, 'fmow/metadata_v2/fmow_train.parquet')
        elif split == 'test':
            metadata_path = os.path.join(root, 'fmow/metadata_v2/fmow_test_gt.parquet')
        else:
            metadata_path = os.path.join(root, os.path.expandvars(split))

        self.df = pd.read_parquet(metadata_path)
        
        # load dataset metainfo
        ds_names = self.sensor_name_mapping.values()
        self.chn_ids = {k: extract_wavemus(load_ds_cfg(k), full_spectra) for k in ds_names } 

        self.normalize = normalize
        self.root = root
        self.min_img_size = min_img_size
        self.max_img_size = max_img_size
        self.keep_sensors = keep_sensors
        self.split = split
        if not self.return_rgb:
            self.df = self.df[self.df['ms_sensor_platform_name'].isin(self.keep_sensors)]
        else:
            self.df = self.df[self.df['rgb_sensor_platform_name'].isin(self.keep_sensors)]
            #filter out all rows where rgb_is_corrupt is True
            # self.df = self.df[self.df['rgb_is_corrupt'] == False]
            logger.info(f'RGB Mode')
        self._subset_df()

        self.log_stats()
        self.problematic_ids = []

        if self.normalize:
            logger.info('Normalizing images')
            self.channelwise_transforms = self._build_ch_transforms()

    def _subset_df(self):
        # remove row in PROBLEMATIC_IDS
        if self.split in self.PROBLEMATIC_IDS:
            # Get problematic IDs only for matching split and channel count
            pids = self.PROBLEMATIC_IDS[self.split].get(self.num_channels, [])
            print(f'problematic_ids for {self.num_channels} channels: {pids}')
            
            if len(pids) > 0:
                len_before = len(self.df)
                self.df = self.df[~self.df['id'].isin(pids)]
                len_after = len(self.df)
                logger.info(f'Removed {len_before - len_after} problematic images for {self.split} split with {self.num_channels} channels')
        self.df = self.df.reset_index(drop=True)


    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            channelwise_transforms[sensor] = transforms.Normalize(self.MEAN[sensor], self.STD[sensor])
        return channelwise_transforms

    def log_stats(self):
        sensor_counts = {sensor: 0 for sensor in self.sensor_name_mapping.keys()}
        for sensor in self.sensor_name_mapping.keys():
            sensor_counts[sensor] = self.df['ms_sensor_platform_name'].apply(lambda x: sensor in x).sum()
        logger.info(f'Dataset size: {self.__len__()}, sensor counts: {sensor_counts}')


    def _load_img(self, path) -> torch.Tensor:
        path = os.path.join(self.root, path)
        return torch.from_numpy(imread(path).astype(self.np_dtype)).permute(2,0,1)

    def _get_label(self, id:str):
        #given a string id, return the index of the label in the CLASS_NAMES list
        try:
            return self.CLASS_NAMES.index(id)
        except ValueError:
            logger.warning(f'Label not found in CLASS_NAMES: {id}')
            return None


    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.return_rgb:
            key = 'rgb_path'
            ds_name, sensor = 'fmow_rgb', 'fmow_rgb'
        else:
            key = 'ms_path'
            ds_name, sensor = self.sensor_name_mapping[row['ms_sensor_platform_name']], self.sensor_name_mapping[row['ms_sensor_platform_name']]
        try: 
            img = self._load_img(row[key])
        except Exception as e:
            faulty_path = os.path.join(self.root, row[key])
            full_traceback = traceback.format_exc()

            logger.info(f'Error loading image: {faulty_path}')
            logger.info(full_traceback)
            # if self.faulty_imgs_file is not None:
            #     with open(self.faulty_imgs_file, 'a') as f:
            #         f.write('\n\n')
            #         f.write('time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
            #         f.write('file: ' + faulty_path + '\n')
            #         f.write(full_traceback)
            self.problematic_ids.append(row['id'])
            return None, None
        chn_id = self.chn_ids[ds_name]

        label_id = '_'.join(row['id'].split('_')[:-2])
        label = self._get_label(label_id)

        if (img.shape[0] == 8 and sensor != 'fmow_wv23') or (img.shape[0] == 4 and sensor != 'fmow_qbge'):
            self.problematic_ids.append(row['id'])
            print(f"[{idx}] Mismatching channel size for {self.split} : {img.shape}, {sensor} | Ids : {len(self.problematic_ids)}")
            
    
        if self.normalize:
            img = self.channelwise_transforms[sensor](img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label




class FmowDataset(BaseDataset):
    def __init__(self, 
            root_dir: str,
            split: str,
            keep_sensors: list = None,
            transform_list = None,
            **kwargs
        ):
        super().__init__('FmowDataset', **kwargs)
        num_channels = len(self.chn_ids) if self.band_ids is None else len(self.band_ids)

        transform = K.AugmentationSequential(*transform_list, data_keys=['image'])

        self.dataset = FmowBenchmarkDataset(
            root=root_dir, 
            split=split, 
            transforms=transform, 
            keep_sensors=keep_sensors,
            num_channels=num_channels, 
            normalize=True
        )

    def _getitem(self, idx):
        return self.dataset[idx]
    
    def _len(self):
        return len(self.dataset)