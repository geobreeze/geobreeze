from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
import logging
from tifffile import imread
import datetime
import traceback
from pathlib import Path
import kornia.augmentation as K
import torch
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.samplers.utils import _to_tuple
from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
from .base import BaseDataset
logger = logging.getLogger()


class ClipMinMaxNormalize(object):
    def __init__(self, min_val=0.0, max_val=10000.):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        clipped = torch.clamp(tensor, self.min_val, self.max_val)
        return (clipped - self.min_val) / (self.max_val - self.min_val)


class SpectralEarthDataset(NonGeoDataset):

    #Mean and Stdev values calculated from 10)%train+val 
    
    ENMAP_MEAN = [
        575.4771411764706, 553.0658647058824, 561.0348529411766, 585.0801764705882, 593.8231058823529, 597.7494117647059, 611.1234823529411, 623.9829647058824, 628.5492647058824, 643.9855176470588,
        651.2347588235294, 654.5498235294117, 658.9776529411764, 670.2240470588235, 677.3810470588235, 675.8211941176471, 679.5913588235294, 691.6472352941176, 693.4439352941176, 708.8818470588236,
        735.6634941176471, 752.1808117647059, 771.4432, 797.4184470588236, 821.5571352941176, 844.5553823529411, 856.5141764705883, 869.6235705882352, 882.626405882353, 890.3532294117647,
        894.1966588235293, 890.7280352941177, 878.3233588235294, 885.7238764705883, 884.0735588235294, 885.535205882353, 894.5968411764705, 901.1996294117647, 903.3418941176469, 902.3626764705883,
        913.3784235294117, 913.5091764705883, 915.7557294117647, 924.2091352941176, 922.6661470588235, 915.9288823529413, 918.1000235294118, 920.433005882353, 922.6426823529412, 931.9387294117648,
        959.5464117647058, 1015.0457529411766, 1122.4451, 1243.1475588235294, 1380.3046529411765, 1524.9906352941175, 1700.1150529411764, 1846.4029352941177, 1965.9293705882355, 2035.8508647058825,
        2124.8800764705884, 2006.9024176470589, 2164.1007588235298, 2201.295764705882, 2217.5599470588236, 2241.1114470588236, 2257.8508941176474, 2293.4513941176474, 2295.942105882353, 2329.9979000000003,
        2326.292376470588, 2349.6223588235293, 2361.6673764705884, 2374.8117117647057, 2402.346282352941, 2410.496964705883, 2431.6017411764706, 2419.9053058823524, 2444.5171647058824, 2531.183852941176,
        2425.5541411764702, 2471.604229411765, 2422.5687882352945, 2317.330111764706, 2574.6946705882356, 2504.359147058824, 2424.882447058824, 2499.579576470588, 2625.260870588235, 2487.760764705882,
        2535.3023588235296, 2347.0476999999996, 2267.6814058823534, 2505.3018529411765, 2318.720676470588, 2535.2417764705883, 2321.4387588235295, 2551.916017647059, 2332.7988764705883, 2498.293535294117,
        2418.3412823529416, 2594.483141176471, 2624.9612470588236, 2660.7274470588236, 2695.2597176470586, 2721.283570588235, 2752.1525588235295, 2785.4055529411767, 2795.832494117647, 2810.323447058824,
        2816.4612941176474, 2724.7353176470588, 2874.6722999999997, 2753.389494117647, 2538.9428352941172, 2540.952776470588, 2531.8915058823527, 2541.6359235294117, 2537.8132941176473, 2545.7771764705885,
        2584.229129411765, 2628.7360352941178, 2659.677841176471, 2686.098517647059, 2736.117676470588, 2728.8450647058826, 1677.5257176470586, 1716.1004705882353, 1756.1821470588234, 1795.0956647058824,
        1832.3942176470587, 1871.6317294117648, 1907.0384294117644, 1945.304764705882, 1982.7070764705882, 2007.6600823529411, 2026.9125705882352, 2045.4060529411763, 2052.236523529412, 2048.052411764706,
        2039.761905882353, 2020.9278176470586, 2000.5440352941177, 1978.051723529412, 1954.4188294117646, 1127.5802235294118, 1206.1269529411763, 1122.9020176470588, 1062.9485117647057, 1072.9123352941178,
        1226.181705882353, 1253.1995647058823, 1272.9365529411764, 1200.8745117647056, 1220.4586176470586, 1221.7800529411766, 1271.926111764706, 1254.0899705882355, 1262.1880823529411, 1271.1464529411767,
        1289.0367117647058, 1279.846294117647, 1303.4488882352941, 1302.2262411764705, 1310.6051235294117, 1312.3007, 1325.088794117647, 1326.060594117647, 1336.033705882353, 1337.7121764705882,
        1338.4857941176472, 1345.899711764706, 1360.058, 1353.2447294117646, 1347.3704588235294, 1303.8918470588235, 1277.8740294117647, 1236.1976764705882, 1212.7242529411765, 1176.6520529411764,
        1171.2157647058823, 1130.1141823529413, 1117.0207294117647, 1087.1962529411765, 1073.7413823529412, 1068.436894117647, 1067.0357411764705, 1028.8896941176472, 1035.6631235294117, 1035.7437,
        1063.7975, 1044.7913176470588, 1069.4804352941176, 1041.1795235294117, 1057.0329705882352, 1014.9592941176471, 1033.215905882353, 958.2710823529411, 996.3860764705881, 930.197994117647,
        955.0627999999999, 804.2681411764706
    ]
    
    ENMAP_STD = [1106.5435370660741, 1098.4775789763778, 1099.754503867262, 1107.9016512741657, 1108.1490166927792, 1102.731408979418, 1105.1273595254195, 1104.1428466215461, 1100.5090914001448, 1103.8449516351534,
                1103.074968072294, 1101.7305220206165, 1103.6929883311498, 1108.156704009022, 1113.180717195585, 1111.3264664576927, 1112.078036425091, 1116.8851242066162, 1115.269206556685, 1118.283110723213,
                1125.1903322650849, 1122.2482375759776, 1118.6523460240207, 1119.7943115972369, 1122.384581426843, 1128.4542641025332, 1130.0474374694766, 1132.2285112608977, 1137.2117332073603, 1143.5918017341787,
                1152.261895293684, 1159.333272480088, 1165.125432679377, 1175.4396505024672, 1178.968968185138, 1184.8198531357962, 1192.7487825519538, 1198.6966680096523, 1202.0697286975385, 1204.5838199989582,
                1211.2421703959606, 1211.3298617948885, 1211.0759603681993, 1218.702830491934, 1224.0703088054038, 1224.4649881462165, 1229.5655615278333, 1235.2226212689918, 1237.7381769074066, 1237.71015662233,
                1249.0416463550127, 1239.6333911143447, 1228.4866619258632, 1215.133092865001, 1217.7665740986365, 1225.5483426138262, 1259.365916061097, 1299.037212315706, 1343.7720459440516, 1366.6551009442667,
                1414.1483646581357, 1374.0831543650456, 1431.2108683281558, 1439.0772246014392, 1441.9192268939676, 1453.1171520212697, 1458.9692842048964, 1475.3954579094013, 1479.2797041373017, 1495.0139896749326,
                1489.5209550383727, 1500.4743240257164, 1504.9890482101453, 1508.6324259794985, 1518.4836969916394, 1514.2801477438647, 1521.954985526914, 1514.3295990270608, 1522.7949069797723, 1502.4981820626954,
                1509.8160461713524, 1482.0943812947858, 1501.3391338557387, 1444.1265787746452, 1519.0767971788869, 1552.8245006862762, 1422.8169014770315, 1636.5552748522418, 1497.3548708871126, 1611.8990738990394,
                1453.395419999223, 1529.4579069648648, 1463.778005952535, 1447.6866167626372, 1456.107963817661, 1470.3466814163921, 1457.506754056099, 1476.4730119157534, 1454.7989079461179, 1459.7262073412087,
                1482.9215029788922, 1492.107795870563, 1503.6233387176185, 1520.0192633038207, 1538.4882826089286, 1552.8701531770341, 1571.2442764782352, 1589.2025714555737, 1592.669303910329, 1594.4052415021245,
                1585.9682795041847, 1512.1188821280996, 1539.2106003940228, 1501.1296161895873, 1406.6221538306936, 1432.9397853596429, 1433.604386134263, 1436.811848761955, 1434.01595237993, 1434.3749815088133,
                1451.4149403893016, 1471.559370299194, 1486.3187234403417, 1491.958018533633, 1504.1775682735238, 1515.4551137823369, 1207.9869621204668, 1216.9650068869603, 1226.7344785501537, 1236.2124604641854,
                1244.720323255137, 1255.584199352594, 1266.9737447682721, 1280.61055587049, 1294.695342732917, 1303.2678570859468, 1308.6742984303105, 1314.801337832619, 1316.224837310629, 1312.1669673514077,
                1306.3328333810568, 1296.884952526696, 1288.6680646515654, 1280.3079363593006, 1270.1416752539417, 1050.0910003551855, 1092.0142411177599, 1025.539385245312, 928.2917587619615, 970.8474304162509,
                1052.628833713246, 1083.1192575354373, 1079.4077976080303, 1027.0369610708683, 1021.3342095354814, 1027.5051503910465, 1048.9566605083207, 1037.242676188447, 1029.3844890209475, 1037.585796524963,
                1037.5284112881482, 1029.6870383198145, 1035.4805113909747, 1037.176576915736, 1026.9866643597047, 1026.0527076007947, 1020.9900955421884, 1022.3332097348529, 1013.282325561588, 1015.2084426821608,
                998.0391648938682, 1006.1312184710196, 1009.778087135915, 1014.8869150286203, 1006.3457439315932, 986.4112525421942, 964.4220219089839, 949.9490449095832, 930.1805187761163, 918.5355755075361,
                907.5492944740072, 888.153296986416, 870.9176846620923, 866.0272514761022, 845.6820550456786, 847.5502846936974, 837.6229640663133, 820.5756250146151, 812.2526596212555, 830.4762830777298,
                848.3608730108356, 847.0815263032866, 850.3598804111441, 848.764620092764, 850.3784520434889, 848.0706088362521, 833.4373715346229, 809.1528182206465, 823.6386440911573, 821.9364678101272,
                825.0898532788261, 782.2638529431425

    ]

    #converts from hierarchical to serial order
    # The following classes are removed ion 19-class version and hence mapped to 43 (remove)
    # 122-142: Road and rail networks and associated land , Port areas Airports ,Mineral extraction sites ,Dump sites ,Construction sites ,Green urban areas ,Sport and leisure facilities s versio:
    # 332, 334: Bare rock, Burnt area
    # 423 : Intertidal flats
    CORINE_CLASSES_43 = {111: 0, 112: 1, 121: 2, 122: 43, 123: 43, 124: 43, 131: 43, 132: 43, 133: 43, 141: 43, 142: 43, 
                  211: 11, 212: 12, 213: 13, 221: 14, 222: 15, 223: 16, 231: 17, 241: 18, 242:19, 243: 20, 244: 21, 
                  311: 22, 312: 23, 313: 24, 321: 25, 322: 26, 323: 27, 324: 28, 331: 29, 332: 43, 333: 31, 334: 43, 335:43,
                  411: 33, 412: 34, 421: 35, 422: 36, 423: 43, 
                  511: 38, 512: 39, 521: 40, 522: 41, 523: 42,
                  999: 43} #999 = NODATA

    label_converter = { #from BEN in torchgeo
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
        43: None,
    }

    RGB_CHANNELS: dict = {
        'spectral_earth': [43, 28, 10],
    }

    def __init__(self,
                 root, 
                 split=None,  #can be one of ['train', 'val', 'test', None]: None is reserved for pretrain only
                 normalize=True, 
                 task_dir='corine', # can be one of ['enmap', 'cdl', 'corine', 'nlcd']
                 return_rgb=False,
                 multilabel=True,
                 transforms=None,
                 return_chns_by_id=None):

        """
        params:
        - split: str, can be one of ['train', 'val', 'test', None]: None is reserved for pretrain only
        - normalize: bool, whether to normalize the images
        - root: str, root directory for the dataset
        - task_dir: str, name of the task directory, can be one of ['enmap', 'cdl', 'corine', 'nlcd']
            - enmap: pretrain
            - cdl: crop type classification
            - corine: land cover classification
            - nlcd: land cover classification
        - return_rgb: bool, whether to return only RGB channels
        - transforms: callable, transforms to apply to the dataset
        - faulty_imgs_file: str, path to a file with faulty images
        - full_spectra: bool, whether to return full spectra
        - multilabel: bool, whether to return a multilabel target
            - if True, the target will be a one-hot encoded tensor of shape (num_classes,)
            - if False, the target will be pixel-wise labels of shape (128,128)
        """

        root = os.path.expandvars(root)
        self.transforms = transforms
        self.return_rgb = return_rgb
        self.normalize = normalize
        if return_chns_by_id is not None:
            self.return_chns_by_id = torch.tensor(return_chns_by_id)
        else:
            self.return_chns_by_id = None

        assert split in ['train', 'val', 'test', None], f"split must be one of ['train', 'val', 'test', None], got {split}"
        assert task_dir in ['enmap', 'cdl', 'corine', 'nlcd'], f"task_dir must be one of ['enmap', 'cdl', 'corine', 'nlcd'], got {task_dir}"

        split = 'train' if split == None else split

        if task_dir == 'enmap' and split != 'train':
            raise ValueError('Enmap pretrain task only supports train split')

        self.splits_dir = os.path.join(root, f'splits/{task_dir}')
        self.data_dir = os.path.join(root, f'enmap')

        if task_dir != 'enmap':
            self.labels_dir = os.path.join(root, f'spectral_earth_downstream_datasets/enmap_{task_dir}/{task_dir}')
            # print(f'Labels dir: {self.labels_dir}')
            # logger.info(f'SE Labels dir: {self.labels_dir}')
            if task_dir == 'corine':
                self.num_classes = 19
                self.data_dir = os.path.join(root, f'spectral_earth_downstream_datasets/enmap_{task_dir}/enmap')
            elif task_dir == 'cdl':
                raise NotImplementedError('CDL task not implemented yet')
            elif task_dir == 'nlcd':
                raise NotImplementedError('NLCD task not implemented yet')

        else:
            self.labels_dir = None

        # read file metainfo
        if split in ['train', 'val', 'test']:
            metadata_path = os.path.join(self.splits_dir, f'{split}.txt')

        #read metadata from a txt file with no commas, but newlines
        with open(metadata_path, 'r') as f:
            self.metadata = f.read().splitlines()

        
        if self.normalize:
            self.channelwise_transforms = self._build_ch_transforms()

        self.multilabel = multilabel


    def _corine_label_43_to_19(self, label):
        return self.label_converter[self.CORINE_CLASSES_43[label]]

    def _build_ch_transforms(self):
        self.MEAN = torch.tensor(self.ENMAP_MEAN)
        self.STD = torch.tensor(self.ENMAP_STD)
        
        return transforms.Compose([
            # ClipMinMaxNormalize(min_val=0.0, max_val=10000.),
            transforms.Normalize(self.MEAN, self.STD),
        ])


    def __len__(self):
            return len(self.metadata)

    def _load_img(self, path) -> torch.Tensor:
        return torch.from_numpy(imread(path).astype('float32')).permute(2,0,1) 


    def _load_label(self, path) -> torch.Tensor:
        if not self.multilabel: # return pixel-wise labels
            return torch.from_numpy(imread(path))#.astype('float32'))
        else: # return one-hot encoded labels
            label = imread(path)
            unique_classes = np.unique(label)
            # print(f'Unique classes: {unique_classes}')
            indices = [self._corine_label_43_to_19(c) for c in unique_classes]
            # print(f'Indices: {indices}')
            # remove any None values
            indices = [i for i in indices if i is not None]
            # print(f'Indices: {indices}')
            image_target = torch.zeros(self.num_classes, dtype=torch.long)
            image_target[indices] = 1
            return image_target


    def __getitem__(self, idx):
        # Load the image
        path = self.metadata[idx]
        img_path = os.path.join(self.data_dir, path)

        try:
            img = self._load_img(img_path)
        except Exception as e:
            # logger.error(f"Error loading image {img_path}: {e}")
            print(f"Error loading image {img_path}: {e}")
            traceback.print_exc()
            img = None

        if self.labels_dir: #i.e. its a downstream task
            label_path = os.path.join(self.labels_dir, path)

            try:
                label = self._load_label(label_path)
            except Exception as e:
                # logger.error(f"Error loading label {label_path}: {e}")  
                print(f"Error loading label {label_path}: {e}")
                traceback.print_exc()
                label =  None

        if self.normalize:
            # print('Normalizing image')
            img = self.channelwise_transforms(img)

        if self.return_rgb:
            rgb_idx = torch.tensor(self.RGB_CHANNELS[self.ds_name])
            img = img[rgb_idx]
        elif self.return_chns_by_id is not None:
            idx = self.return_chns_by_id
            img = img[idx]
        else:
            pass

        assert self.labels_dir is not None, "Labels dir is not set"

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
        

    
    def get_targets(self):
        return np.arange(self.num_classes)




class CorineDataset(BaseDataset):

    def __init__(self,
            root: str,
            split: str,
            transform_list: list = [],
            normalize: bool = True,
            **kwargs
        ):
        super().__init__('corine', **kwargs)

        trf = K.AugmentationSequential(*transform_list, data_keys=['image'])

        self.dataset = SpectralEarthDataset(
            root=root, split=split, task_dir='corine', transforms=trf, normalize=normalize
        )

    def _getitem(self, idx):
        return self.dataset[idx]
    
    def _len(self):
        return len(self.dataset)
