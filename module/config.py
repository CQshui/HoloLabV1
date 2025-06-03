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

    file_info: Dict[str, Any] = field(default_factory=lambda: {
        'holo_type'     : "Off-Axis",
        'holo_type_list': ['Inline', 'Off-Axis'],
        'image_path'    : r'E:\Projects\HoloLab\test_data\hologram',
        'image_name'    : 'Image__2024-04-26__20-13-40.bmp',
    })

    image_info: Dict[str, Any] = field(default_factory=lambda: {
        'pixel_size'        : 0.098,   # um
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
    pre_process: Dict[str, Any] = field(default_factory=lambda: {
        'method'            : 'Subtraction',
        'method_list'       : ['None', 'Subtraction', 'AVG-Subtraction','FFT', 'AI'],
        'coeff'             : 0.5,
        'reserve_1'         : 'None',
        'background_path'   : r'E:\Projects\HoloLab\test_data/preprocessing',
        'background_name'   : 'inline_multi_particle.jpg',
        'model_path'        : r'E:\Projects\HoloLab/models/pre_process',
        'model_name'        : 'model_name'
    })

    spectrum: Dict[str, Any] = field(default_factory=lambda: {
        'method'            : 'Manual_Select',
        'method_list'       : ['None', 'Manual_Select', 'Give_Values', 'Auto_Define', 'FFT', 'AI'],
        'center_mask_radius': 100,
        'threshold'         : 180,
        'reserve_1'         : 'None',
        'model_path'        : r'E:\Projects\HoloLab/models/spectrum',
        'model_name'        : 'model_name',
        'ROI_rectangle'     : {
            'center_x'      : 50,
            'center_y'      : 50,
            'rect_width'    : 50,
            'rect_height'   : 50
        },
    })

    reconstruction: Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'Angular_GPU',
        'method_list'   : ['Angular_CPU', 'Angular_GPU'],
        'reserve_1'     : 'None',
        'model_path'    : r'E:\Projects\HoloLab/models/reconstruction',
        'model_name'    : 'model_name',
        'cpu_num'       : 1,
        'gpu_num'       : 1,
        'z_start'       : 0.15,  # 实际需要 * unit_mm
        'z_end'         : 0.20, # 实际需要 * unit_mm
        'z_step'        : 0.01   # 实际需要 * unit_mm
    })

    focusing: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'AI_Wavelet',
        'method_list': ['Wavelet', 'Gradient', 'AI', 'AI_Wavelet', 'AI_Gradient'],
        'yolo_model_path': r'E:\Projects\HoloLab\models\yolo_detection.pth',
        'rcf_model_path': r'E:\Projects\HoloLab\models\rcf_edge_detection.pth',
        'cpu_num': 1,  # cpu线程数
        'gpu_id': 0,
        'device': 'cpu',
        'device_list': ['cuda', 'cpu'],
        'rcf_scale': 8,  # rcf所处理图像的缩放倍率，图像原尺寸要/rcf_scale
        'batch_root': r'F:\liujianli\data\experiment_data\20250424\results',
        # 批量处理：根目录，下有每个图像对应子文件夹，子文件夹内部有reconstruction文件夹存放重建图像
        'get_model': False,  # 批量处理：如果为True，将直接给Focus类传入模型本身，而不是根据路径加载模型
    })

    segmentation: Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI',
        'method_list'   : ['Angular Spectrum', 'Fresnel', 'AI'],
        'gray_thresh'   : 127,
        'block_size'    : 32,
        'reserve_1'     : 'None',
        'model_path'    : r'E:\Projects\HoloLab/models/segmentation',
        'model_name'    : 'model_name',
    })

    identification: Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI',
        'method_list'   : ['Angular Spectrum', 'Fresnel', 'AI'],
        'type'          : 'Nisha',
        'type_dict'     :{
            'Nisha'     : ['AAA', 'BBB', 'CCC'],
            'Pla'       : ['AAA', 'BBB', 'CCC'],
            'GL'        : ['AAA', 'BBB', 'CCC']
        },
        'reserve_1'     : 'None',
        'model_path'    : r'E:\Projects\HoloLab/models/identification',
        'model_name'    : 'model_name',
    })

    phase: Dict[str, Any] = field(default_factory=lambda: {
        'method'        : 'AI',
        'method_list'   : ['Angular Spectrum', 'Fresnel', 'AI'],
        'reserve_1'     : 'None',
        'model_path'    : r'E:\Projects\HoloLab/models/phase',
        'model_name'    : 'model_name'
    })

    data_summary: Dict[str, Any] = field(default_factory=lambda: {
        'density'   : 'None',
        'model_path': r'E:\Projects\HoloLab/models/data_analysis',
        'model_name': 'model_name',
        'show_size_distribution'    : True,
        'show_concentration_time'   : False,
        'show_R50_distribution'     : True,
        'show_R90_distribution'     : False,
        'show_R200_distribution'    : False,
    })

    save_and_load: Dict[str, Any] = field(default_factory=lambda: {
        'config_save_path'      : r'E:\Projects\HoloLab/results',
        'config_save_name'      : 'config.json',
        'config_load_path'      : r'E:\Projects\HoloLab/results',
        'config_load_name'      : 'config_.json',

        'data_load_path'        : r'E:\Projects\HoloLab/results',
        'data_load_name'        : 'data___50.npz',
        'data_NPZ_only'         : True,

        'save_to_disk'          : False, # 在处理的时候就保存
        'creat_sub_dir'         : True, # 保存时：以图片名字创建子文件夹

        'data_save_method'      : 'NPZ Data',
        'data_save_method_list' : ['NPZ Data', 'Image - BMP', 'Image - PNG', 'Image - JPG'],
        'data_save_path'        : r'E:\Projects\HoloLab/results',
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

    operation_options: Dict[str, Any] = field(default_factory=lambda: {
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

    multi_processing: Dict[str, Any] = field(default_factory=lambda: {
        'images_path'            : r'E:\Projects\HoloLab/test_data/hologram',
        'images_name_suffix'     : 'any',
        'images_name_suffix_list': ['any', 'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff'],
        'max_handle_num'         : None,
        'images_urls'           : [],
        'run_open_image'        : True,
        'run_preprocessing'     : True,
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