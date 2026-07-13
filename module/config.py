from common.constants import unit_cm, unit_mm, unit_um, unit_nm
from dataclasses import dataclass, field
import json
from typing import Dict, Any

@dataclass
class CameraConfig:
    wavelength: float = 532e-9
    pixel_size: float = 4.5e-6
    image_height: int = 1024
    image_width: int = 1024

@dataclass
class HoloConfig:
    ''''''
    '''图像参数_________________________________________'''

    file_info:              Dict[str, Any] = field(default_factory=lambda: {
        'holo_type'     : "Inline",
        'holo_type_list': ['Inline', 'Off-Axis'],
        'image_path'    : r'E:\Projects\HoloLabV1\test_data/hologram',
        'image_name'    : 'inline_multi_particle.jpg'
        # 'read_mode'     : 'Single Image',
        # 'read_mode_list': ['Single Image', 'Multi Images']
    })

    '''相机参数_________________________________________'''
    camera: Dict[str, Any] = field(default_factory=lambda: {
        'camera_name': 'No Camera',
        'camera_type': 'Regular',
        'camera_type_list': ['Regular', 'Polar'],
        'sensor_size': 'N/A',
        # 'pixel_size'        : 3.45,

        'exposure_time': 2000,
        'frame_rate': 5,
        'gain': 0,
        'image_width': 1000,
        'image_height': 1000,
        'offset_x': 0,
        'offset_y': 0,
        'center_x': True,
        'center_y': True,

        'num_to_save': 5,
        'save_path': 'D:/Development/HoloLab/test_data/camera',
        'record_mode': 'single',
        'record_mode_list': ['single', 'multiple'],
        'save_format': 'jpg',
        'save_format_list': ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff'],
        'jpg_quality': 100,

        'enable_balance_white': False,  # 是否设置白平衡
        'enable_ultrashort_exposure': False,  # 是否设置超短曝光
        'enable_gain_mode': False,  # 是否设置增益模式
    })

    image_info:             Dict[str, Any] = field(default_factory=lambda: {
        'pixel_size'        : 5.0,   # um
        'wavelength'        : 532.0,   # nm
        'pixel_num_x'       : 'None',
        'pixel_num_y'       : 'None',
        'off_axis_angle_x'  : 0.0,  # 与Z轴的夹角（离轴图参数）
        'off_axis_angle_y'  : 0.0,  # 与Z轴的夹角（离轴图参数）
        'ROI_rectangle'     : {
            'center_x'      : 50,
            'center_y'      : 50,
            'rect_width'    : 50,
            'rect_height'   : 50
        },
    })

    '''处理参数_________________________________________'''
    pre_process:            Dict[str, Any] = field(default_factory=lambda: {
        'method'            : 'Subtraction',
        'method_list'       : ['None', 'Subtraction', 'AVG-Subtraction','FFT', 'AI'],
        'coeff'             : 0.5,
        'reserve_1'         : 'None',
        'background_path'   : r'E:\Projects\HoloLabV1\test_data/preprocessing',
        'background_name'   : 'inline_multi_particle.jpg',
        'model_path'        : r'E:\Projects\HoloLabV1\test_data/pre_process',
        'model_name'        : 'model_name'
    })

    polarization:           Dict[str, Any] = field(default_factory=lambda: {
        'split_image'       : True,  # 分离四个偏振态
        'split_mode'        : 'quadrant',
        'split_mode_list'   : ['quadrant', 'super_pixel'],

        'polar_coeff'       : False,  # 计算高阶偏振量
        'device'            : 'cpu',
        'device_list'       : ['gpu', 'cpu'],
    })

    spectrum:               Dict[str, Any] = field(default_factory=lambda: {
        'method'            : 'Manual_Select',
        'method_list'       : ['Manual_Select', 'Give_Values', 'Auto_Define'],
        'center_mask_radius': 100,
        'threshold'         : 180,
        'ROI_rectangle'     : {
            'center_x'      : 50,
            'center_y'      : 50,
            'rect_width'    : 50,
            'rect_height'   : 50
        },
        'model_path'        : 'E:\Projects\HoloLabV1\models/spectrum',
        'model_name'        : 'model_name'
    })

    reconstruction:         Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'Angular_GPU',
        'method_list'   : ['Angular_CPU', 'Angular_GPU', 'Fresnel', 'AI'],
        'model_path'    : 'E:\Projects\HoloLabV1\models/reconstruction',
        'model_name'    : 'model_name',
        'cpu_num'       : 10,
        'gpu_num'       : 1,
        'z_start'       : 40.0,     # 实际需要 * unit_mm
        'z_end'         : 100.0,     # 实际需要 * unit_mm
        'z_step'        : 1.00     # 实际需要 * unit_mm
    })

    focusing_BackUP:        Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI_Gradient',
        'method_list'   : ['Wavelet', 'Gradient', 'AI', 'AI_Wavelet', 'AI_Gradient'],
        'reserve_1'     : 'None',
        'model_path'    : 'E:\Projects\HoloLabV1\models/focusing',
        'model_name'    : 'model_name',
        'cpu_num'       : 1,
        'gpu_num'       : 1
    })
    focusing:               Dict[str, Any] = field(default_factory=lambda: {
        'method'            : 'AI_Gradient',
        'method_list'       : ['AI', 'Wavelet', 'Gradient', 'AI_Wavelet', 'AI_Gradient'],
        'yolo_model_path'   : r'E:\Projects\HoloLabV1\models/focusing/yolo_detection.pth',
        'rcf_model_path'    : r'E:\Projects\HoloLabV1\models/focusing/rcf_edge_detection.pth',
        'rcf_scale'         : 2,  # rcf所处理图像的缩放倍率，图像原尺寸要/scale
        'device'            : 'cuda',
        'device_list'       : ['cuda', 'cpu'],
        'cpu_num'           : 8,
        'gpu_id'            : 0,
        'batch_root'        : r'E:\Projects\HoloLabV1\test_data\batch test',
        # 批量处理：根目录，下有每个图像对应子文件夹，子文件夹内部有reconstruction文件夹存放重建图像
        'get_model'         : False  # 批量处理：如果为True，将直接给Focus类传入模型本身，而不是根据路径加载模型
    })

    segmentation_BackUP:    Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'Deep_Learning',
        'method_list'   : ['Global_Threshold', 'Adaptive_Threshold', 'Deep_Learning'],
        'gray_thresh'   : 127,
        'block_size'    : 32,
        'model_path'    : 'E:\Projects\HoloLabV1\models/segmentation',
        'model_name'    : 'model_name',
    })
    segmentation:           Dict[str, Any] = field(default_factory=lambda: {
        'type'          : 'holo',
        'type_list'     : ['holo', 'polar'],
        'method'        : 'Deep_Learning',
        'method_list'   : ['Deep_Learning', 'Global_Threshold', 'Adaptive_Threshold'],
        'gray_thresh'   : 100,
        'block_size'    : 32,
        'device'        : 'cpu',  # 'cuda' or 'cpu'
        'device_list'   : ['cuda', 'cpu'],
        'cpu_num'       : 8,
        'gpu_num'       : 1,
        'model_path'    : r'E:\Projects\HoloLabV1\models/segmentation/78_iou_0.9460_F1_0.9719.pth',
        'model_name'    : 'segmentation',
        'output_dir'    : r'E:\Projects\HoloLabV1/results'
    })

    identification_BackUP:  Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI',
        'method_list'   : ['Angular Spectrum', 'Fresnel', 'AI'],
        'type'          : 'Nisha',
        'type_dict'     :{
            'Nisha'     : ['AAA', 'BBB', 'CCC'],
            'Pla'       : ['AAA', 'BBB', 'CCC'],
            'GL'        : ['AAA', 'BBB', 'CCC']
        },
        'reserve_1'     : 'None',
        'model_path'    : 'E:\Projects\HoloLabV1/models/identification',
        'model_name'    : 'model_name',
    })
    identification:         Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'onlyholo',
        'method_list'   : ['onlyholo', 'polar'],
        'type'          : 'Nisha',
        'type_dict'     :{
            'Nisha'     : ['gaoyingshi', 'nachangshi', 'shiyingshi', 'yilishi'],
            'Pla'       : ['AAA', 'BBB', 'CCC'],
            'GL'        : ['AAA', 'BBB', 'CCC']
        },
        'model_path'    : r'E:\Projects\HoloLabV1/models/identification/gao,na,shi,yi/amplitude',
        'model_name'    : '2025.0407_gao,na,shi,yi.pth',
    })

    phase:                  Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI',
        'method_list'   : ['Angular Spectrum', 'Fresnel', 'AI'],
        'reserve_1'     : 'None',
        'model_path'    : 'E:\Projects\HoloLabV1/models/phase',
        'model_name'    : 'model_name'
    })

    data_summary_BackUP:    Dict[str, Any] = field(default_factory=lambda: {
        'density'   : 'None',
        'model_path': 'E:\Projects\HoloLabV1/models/data_analysis',
        'model_name': 'model_name',
        'show_size_distribution'    : True,
        'show_concentration_time'   : False,
        'show_R50_distribution'     : True,
        'show_R90_distribution'     : False,
        'show_R200_distribution'    : False,
    })
    data_summary:           Dict[str, Any] = field(default_factory=lambda: {
        'density'       : {'gaoyingshi':4.5, 'nachangshi':2.6, 'shiyingshi': 2.6, 'yilishi':2.6},
        'method'        : 'onlyholo',
        'method_list'   : ['onlyholo', 'polar'],
        'type'          : 'Nisha',
        'type_dict'     : {
            'Nisha': ['gaoyingshi', 'nachangshi', 'shiyingshi', 'yilishi'],
            'Pla': ['AAA', 'BBB', 'CCC'],
            'GL': ['AAA', 'BBB', 'CCC']
        },
        'save_size'         : True,
        'save_concentration': True,
        'save_category'     : True
    })

    save_and_load:          Dict[str, Any] = field(default_factory=lambda: {
        'config_save_path'      : 'E:\Projects\HoloLabV1/results',
        'config_save_name'      : 'config.json',
        'config_load_path'      : 'E:\Projects\HoloLabV1/results',
        'config_load_name'      : 'config_.json',

        'data_load_path'        : 'E:\Projects\HoloLabV1/results',
        'data_load_name'        : 'data___50.npz',
        'data_NPZ_only'         : True,

        'save_to_disk'          : False, # 在处理的时候就保存
        'creat_sub_dir'         : True, # 保存时：以图片名字创建子文件夹

        'data_save_method'      : 'NPZ Data',
        'data_save_method_list' : ['NPZ Data', 'Image - BMP', 'Image - PNG', 'Image - JPG'],
        'data_save_path'        : 'E:\Projects\HoloLabV1/results',
        'data_save_name'        : 'data.npz',
        'save_all'              : False,
        'save_raw_hologram'     : True,
        'save_pre_process'      : True,
        'save_polarization'     : True,
        'save_spectrum'         : True,
        'save_reconstruction'   : True,
        'save_focusing'         : True,
        'save_segmentation'     : True,
        'save_identification'   : True,
        'save_phase'            : True,
        'save_data_summary'     : True
    })

    operation_options:      Dict[str, Any] = field(default_factory=lambda: {
        'run_open_image'       : True,
        'run_preprocessing'    : False,
        'run_polarization'     : False,
        'run_spectrum'         : True,
        'run_reconstruction'   : True,
        'run_focusing'         : True,
        'run_segmentation'     : False,
        'run_identification'   : False,
        'run_phase'            : False,
        'run_data_summary'     : False
    })

    operator_model_load:    Dict[str, Any] = field(default_factory=lambda: {          # todo, 景深拓展批量处理时，不重复加载模型
        'focusing'          : None,
    })

    multi_processing:       Dict[str, Any] = field(default_factory=lambda: {
        'images_path'            : 'E:\Projects\HoloLabV1/test_data/batch test',
        'images_name_suffix'     : 'any',
        'images_name_suffix_list': ['any', 'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff'],
        'max_handle_num'         : None,
        'images_urls'           : [],
        'containt_non_image'    : False,
        'non_images_urls'       : [],
        'run_open_image'        : True,
        'run_preprocessing'     : False,
        'run_polarization'      : False,
        'run_spectrum'          : True,
        'run_reconstruction'    : True,
        'run_focusing'          : True,
        'run_segmentation'      : True,
        'run_identification'    : True,
        'run_phase'             : False,
        'run_data_summary'      : True
    })

    def save_config(self, path: str) -> None:
        """保存配置到JSON文件"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=4)

            msg = f"Save Config to: {path}"
            # print(f"配置文件已保存到：{path}")

        except Exception as e:
            msg = f"Failed to save Config: {e}"
            # print(f"保存配置失败: {e}")

        return msg

    def load_config(self, path: str) -> None:
        """从JSON文件加载配置并更新当前实例"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            msg = f"Load Config from: {path}"

        except Exception as e:
            msg = f"Failed to load Config: {e}"
            # print(f"更新配置失败: {e}")

        return msg

    @classmethod
    def creat_config(cls, path: str) -> 'HoloConfig':
        """从JSON文件加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return cls()  # 返回默认配置

if __name__ == "__main__":

    '方式1：直接实例创建'
    holo_config = HoloConfig()
    print(holo_config.file_info['save_to_disk'])

    holo_config.load_config('config_000.json')
    print(holo_config.file_info['save_to_disk'])

    '方式1：从JSON文件加载创建'
    holo_config2 = HoloConfig.creat_config("config.json")
    pre_process = str(holo_config2.pre_process).replace(",", '\n')
    print(f"pre_process: \n{pre_process}")