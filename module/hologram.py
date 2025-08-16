import os
import cv2
import numpy as np
import pickle

from common.constants import unit_cm, unit_mm, unit_um, unit_nm
from module.config import HoloConfig

'''不要了 接受 key_config 字典'''
class Hologram_using_Key_Config:
    def __init__(self, key_config=None):
        ''''''
        self.key_config = key_config if key_config is not None else {'Empty': None}

        self.main_hologram_infos(self.key_config)
        self.main_wave_parameter(self.key_config)
        self.main_process_config(self.key_config)
        self.main_data_analysis(self.key_config)

    '''光学系统参数-------------------------------------------------'''
    def main_optical_config(self, key_config):
        a = 1

        # self.z_start        = key_config.get('z_start'   , unit_mm *  0.0)
        # self.z_end          = key_config.get('z_end'     , unit_mm * 10.0)
        # self.z_step         = key_config.get('z_step'    , unit_mm *  1.0)
        # self.z_num          = int(np.round((self.z_end - self.z_start) / self.z_step)) + 1
        # self.z_array        = np.linspace(self.z_start, self.z_end, num=  self.z_num)

    '''全息图像参数-------------------------------------------------'''
    def main_hologram_infos(self, key_config):

        self.hologram_type      = key_config.get('hologram_type', 'inline')  # inline, offaxis
        self.image_path         = key_config.get('image_path', None)         # r'./images_data'
        self.image_name         = key_config.get('image_name', None)
        self.save_path          = None
        self.save_to_disk       = True
        self.creat_sub_dir      = True
        self.pixel_size         = key_config.get('pixel_size', unit_um * 5.0)
        self.wavelength         = key_config.get('wavelength', unit_nm * 532.0)

        '''全息图 : 读取的图像 hologram_raw、实际处理的变量 hologram'''
        if self.image_path is not None:
            self.hologram_raw   = cv2.imread(os.path.join(self.image_path, self.image_name), cv2.IMREAD_GRAYSCALE)
            self.hologram       = self.hologram_raw.copy()
        else:
            self.hologram_raw   = np.zeros((10, 10), dtype="uint8")
            self.hologram       = np.zeros((10, 10), dtype="uint8")

        '''背景图（如有）_____________________________________________'''
        self.hologram_blank     = np.zeros(self.hologram.shape) # 背景图，如有

        '''偏振全息图像的子图（如有）_________________________________'''
        self.hologram_p045      = self.hologram
        self.hologram_p090      = self.hologram
        self.hologram_p135      = self.hologram
        self.hologram_p180      = self.hologram

        '''图像参数___________________________________________________'''
        self.pixel_num_x        = self.hologram.shape[1]    # 图片宽width
        self.pixel_num_y        = self.hologram.shape[0]    # 图片高height
        self.off_axis_angle_x   = 0                         # 与 Z 轴的夹角，离轴图参数，如有
        self.off_axis_angle_y   = 0                         # 与 Z 轴的夹角，离轴图参数，如有

        # 裁剪参数（如有）
        self.hologram_crop_rect = {
                'center_x'      : 50,
                'center_y'      : 50,
                'rect_width'    : 50,
                'rect_height'   : 50
            }

    '''光波计算参数-------------------------------------------------'''
    def main_wave_parameter(self, key_config):
        ''''''
        '''全息图频谱_________________________________________________'''
        self.hologram_spectrum      = np.zeros(self.hologram.shape, dtype="complex")    # 行（高 height）、列（宽 width）
        self.hologram_spectrum_rect = {
                'center_x'      : 50,
                'center_y'      : 50,
                'rect_width'    : 50,
                'rect_height'   : 50
            }
        self.hologram_spectrum_side = self.hologram_spectrum                            # 离轴中框选出来的频谱，如有

        '''初始波前、波前_____________________________________________'''
        self.wave_front_ini         = np.zeros(self.hologram.shape, dtype="complex")    # 行（高 height）、列（宽 width）
        self.wave_front             = np.zeros(self.hologram.shape, dtype="complex")

        '''重建图簇（多张）___________________________________________'''
        self.reconstruction         = np.zeros(self.hologram.shape, dtype="complex")    # 多张，可以是一个list
        self.reconstruction_each    = [] # 每个颗粒的重建子图列表，可以是一个list

        '''聚焦图、子图、深度图_______________________________________'''
        self.focusing               = np.zeros(self.hologram.shape)
        self.focusing_each          = []
        self.focusing_depth_map     = np.zeros(self.hologram.shape)

        '''切割图、子图_______________________________________________'''
        self.segmentation           = np.zeros(self.hologram.shape)
        self.segmentation_each      = [] # 每个颗粒的切割子图列表，可以是一个list

        '''分类图、分类切割子图_______________________________________'''
        self.identification         = np.zeros(self.hologram.shape)
        self.identification_each    = [] # 每个颗粒的分类子图列表，可以是一个list

        '''相位分析___________________________________________________'''
        self.phase                  = np.zeros(self.hologram.shape)
        self.phase_unwrapped        = np.zeros(self.hologram.shape)
        self.phase_compensated      = np.zeros(self.hologram.shape)
        self.phase_compensation_mat = np.zeros(self.hologram.shape)
        self.phase_corrected        = np.zeros(self.hologram.shape)

        '''窗口函数___________________________________________________'''
        self.window_type            = 'hamming'  # hamming, hann, turkey
        self.window_function        = np.ones(self.hologram.shape)

    '''算法处理方式-------------------------------------------------'''
    def main_process_config(self, key_config):

        self.config_pre_process    = {
            'method'     : None,
            'coeff'      : 0.5,
            'reserve_1'  : None,
            'model_path' : None,
            'background_path': None,
            'background_name': None
        }

        self.config_spectrum       = {
            'method': None,
            'reserve_1'  : None,
            'model_path' : None
        }

        self.config_reconstruction = {
            'method'        : None,
            'reserve_1'     : None,
            'model_path'    : None,
            'z_start'       : unit_mm * 0.0,
            'z_end'         : unit_mm * 10.0,
            'z_step'        : unit_mm * 1.0
        }

        self.config_focusing       = {
            'method': None,
            'reserve_1'  : None,
            'model_path' : None
        }

        self.config_segmentation   = {
            'method': None,
            'reserve_1'  : None,
            'model_path' : None
        }

        self.config_identification = {
            'method'     : None,
            'reserve_1'  : None,
            'model_path' : None
        }

        self.config_phase          = {
            'method'     : None,
            'reserve_1'  : None,
            'model_path' : None
        }

        self.config_data_analysis  = {
            'density'       : None,
            'model_path'    : None
            }

    '''定量分析参数-------------------------------------------------'''
    def main_data_analysis(self, key_config):
        '''
            这里的所有参数，根据实际情况来确定
            - 想法是：让所有的统计分析参数也关联到当前的图像
        '''

        self.particle_size  = 0       # 所有颗粒大小，单位m
        self.particle_num   = 0       # 所有颗粒大小，单位m

        self.AOP            = []      # 所有颗粒的AOP，单位°
        self.DOLP           = []      # 所有颗粒的D值，单位m

def Hologram_using_Key_Config_main():
        key_config = {
            'image_path'    : None,         # r'D:\Development\HoloLab'
            'image_name'    : None,         # r'Temp_inline.bmp'
            'hologram_type' : 'inline',     # inline, offaxis
            'pixel_size'    : unit_um * 5,
            'wavelength'    : unit_nm * 532,
            'z_start'       : unit_mm * 0,
            'z_end'         : unit_mm * 100,
            'z_step'        : unit_mm * 10
        }

        test = Hologram_(key_config)

        print('Hologram is ready.')

'''接受 配置类实例'''
class Hologram:
    def __init__(self, config=None):
        ''''''
        self.config = config if config is not None else HoloConfig()

        self.main_hologram_infos(self.config)
        self.main_wave_parameter(self.config)
        self.main_data_analysis(self.config)

        self.status_msg = 'Ready'

    '''全息图像-------------------------------------------------'''
    def main_hologram_infos(self, config):
        ''''''
        '''全息图 : 读取的图像 hologram_raw、实际处理的变量 hologram'''
        self.hologram_raw   = self._make_empty_image()
        self.hologram       = self._make_empty_image()
        self._image_loaded  = False

        '''全息图 : 读取的图像 hologram_raw、实际处理的变量 hologram'''
        # image_path = self.config.file_info.get('image_path', None)
        # image_name = self.config.file_info.get('image_name', None)
        # self._image_loaded = False
        # if image_path is not None:
        #     try:
        #         self.hologram_raw   = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
        #         self.hologram       = self.hologram_raw.copy()
        #         self._image_loaded   = True
        #     except Exception as e:
        #         if not self._image_loaded:
        #             self.hologram_raw   = self._make_empty_image()
        #             self.hologram       = self._make_empty_image()
        #
        # else:
        #     self.hologram_raw   = self._make_empty_image()
        #     self.hologram       = self._make_empty_image()

        '''背景图（如有）_____________________________________________'''
        self.hologram_blank = self._make_empty_image() # 背景图，如有

        '''偏振全息图像的子图（如有）_________________________________'''
        self.hologram_p000  = self.hologram
        self.hologram_p045  = self.hologram
        self.hologram_p090  = self.hologram
        self.hologram_p135  = self.hologram

        '''图像参数___________________________________________________'''
        self.config.image_info['pixel_num_x']        = self.hologram.shape[1]    # 图片宽width
        self.config.image_info['pixel_num_y']        = self.hologram.shape[0]    # 图片高height
        self.config.image_info['off_axis_angle_x']   = 0                         # 与 Z 轴的夹角，离轴图参数，如有
        self.config.image_info['off_axis_angle_y']   = 0                         # 与 Z 轴的夹角，离轴图参数，如有

    def _make_empty_image(self, shape=(60, 60), gray_value = 200):
        """生成带'Empty'文字的灰度图像"""

        image = cv2.cvtColor(np.zeros(shape, dtype="uint8"), cv2.COLOR_GRAY2BGR)
        text = "Holo LAB"# "Empty"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 计算居中位置
        (w, h), _ = cv2.getTextSize(text, font, 0.4, 1)
        x, y = (shape[1] - w) // 2, (shape[0] + h) // 2

        # 添加文字并转回灰度
        cv2.putText(image, text, (x, y), font, 0.4, (gray_value, gray_value, gray_value), 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''处理图像-------------------------------------------------'''
    def main_wave_parameter(self, config):
        ''''''
        '''全息图频谱_________________________________________________'''
        self.spectrum_raw           = self._make_empty_image()       # 离轴中框选出来的频谱，如有
        self.spectrum               = self._make_empty_image()      # 行（高 height）、列（宽 width）

        '''初始波前、波前_____________________________________________'''
        self.wave_front_ini         = self._make_empty_image()      # np.zeros(self.hologram.shape, dtype="complex")
        self.wave_front             = self._make_empty_image()      # np.zeros(self.hologram.shape, dtype="complex")

        '''重建图簇（多张）___________________________________________'''
        self.reconstruction         = {
            'Z 0': self._make_empty_image(),
            'Z 1': self._make_empty_image() // 2
        }
        self.reconstruction_each    = [] # 每个颗粒的重建子图列表，可以是一个list

        '''聚焦图、子图、深度图_______________________________________'''
        self.focusing               = self._make_empty_image()
        self.focusing_each          = {}
        self.focusing_depth_map     = self._make_empty_image()

        '''切割图、子图_______________________________________________'''
        self.segmentation           = self._make_empty_image()
        self.segmentation_each      = {} # 每个颗粒的切割子图列表，可以是一个list

        '''分类图、分类切割子图_______________________________________'''
        self.identification         = self._make_empty_image()
        self.identification_each    = {} # 每个颗粒的分类子图列表，可以是一个list

        '''相位分析___________________________________________________'''
        self.phase                  = self._make_empty_image()
        self.phase_unwrapped        = self._make_empty_image()
        self.phase_compensated      = self._make_empty_image()
        self.phase_compensation_mat = self._make_empty_image()
        self.phase_corrected        = self._make_empty_image()

        '''窗口函数___________________________________________________'''
        self.window_type            = 'hamming'  # hamming, hann, turkey
        self.window_function        = self._make_empty_image()

    '''定量分析-------------------------------------------------'''
    def main_data_analysis(self, config):

        self.particle_information    = []
        self.particle_total          = 0
        self.concentration           = 0
        self.particle_identification = []

        self.figure_diameter         = None
        self.figure_classification   = None

        self.particle_size  = 0       # 所有颗粒大小，单位m
        self.particle_num   = 0       # 所有颗粒大小，单位m

        self.AoP            = []      # 所有颗粒的AOP，单位°
        self.DoLP           = []      # 所有颗粒的D值，单位m
        self.aop_each       = {}
        self.dolp_each      = {}

        self.polar_S0       = []
        self.polar_S1       = []
        self.polar_S2       = []
        self.polar_amp      = []

    def save_data(self):
        try:
            data_path = self.config.save_and_load['data_save_path']
            data_name = self.config.save_and_load['data_save_name']
            data_url = os.path.join(data_path, data_name)
            os.makedirs(data_path, exist_ok=True)

            '# 收集所有需要保存的变量'
            save_dict = {}

            '# 1. 保存全息图像相关数据'
            hologram_vars = [
                'hologram_raw', 'hologram', 'hologram_blank',
                'hologram_p000','hologram_p045', 'hologram_p090', 'hologram_p135',
                'hologram_spectrum', 'hologram_spectrum_side'
            ]
            for var in hologram_vars:
                if hasattr(self, var):
                    save_dict[var] = getattr(self, var)

            '# 2. 保存波前和重建数据'
            wave_vars = [
                'wave_front_ini', 'wave_front',
                'reconstruction', 'reconstruction_each',
                'focusing', 'focusing_each', 'focusing_depth_map',
                'segmentation', 'segmentation_each',
                'identification', 'identification_each'
            ]
            for var in wave_vars:
                if hasattr(self, var):
                    save_dict[var] = getattr(self, var)

            '# 3. 保存相位分析数据'
            phase_vars = [
                'phase', 'phase_unwrapped', 'phase_compensated',
                'phase_compensation_mat', 'phase_corrected'
            ]
            for var in phase_vars:
                if hasattr(self, var):
                    save_dict[var] = getattr(self, var)

            '# 4. 其他数据'
            other_vars = [
                'window_type', 'window_function'
            ]
            for var in other_vars:
                if hasattr(self, var):
                    save_dict[var] = getattr(self, var)

            '# 5. 保存定量分析数据'
            analysis_vars = [
                'particle_size', 'particle_num', 'AOP', 'DOLP'
            ]
            for var in analysis_vars:
                if hasattr(self, var):
                    save_dict[var] = getattr(self, var)

            '# 6. 保存配置信息（使用pickle序列化）'
            save_dict['config'] = pickle.dumps(self.config)

            '# 保存到npz文件'
            np.savez(data_url, **save_dict)

            msg = f"Data Saved to {data_url}"
        except Exception as e:
            msg = f"Fail to Save Data: {str(e)}"

        return msg

    def load_data(self):
        try:
            data_path = self.config.save_and_load['data_load_path']
            data_name = self.config.save_and_load['data_load_name']

            data_url = os.path.join(data_path, data_name)
            data     = np.load(data_url, allow_pickle=True)

            '# 1. 恢复全息图像相关数据'
            hologram_vars = [
                'hologram_raw', 'hologram', 'hologram_blank',
                'hologram_p000','hologram_p045', 'hologram_p090', 'hologram_p135',
                'hologram_spectrum', 'hologram_spectrum_side'
            ]
            for var in hologram_vars:
                if var in data:
                    setattr(self, var, data[var])

            '# 2. 恢复波前和重建数据'
            wave_vars = [
                'wave_front_ini', 'wave_front',
                'reconstruction', 'reconstruction_each',
                'focusing', 'focusing_each', 'focusing_depth_map',
                'segmentation', 'segmentation_each',
                'identification', 'identification_each'
            ]
            for var in wave_vars:
                if var in data:
                    # 处理字典类型的变量
                    if var in ['reconstruction']:
                        setattr(self, var, data[var].item())
                    else:
                        setattr(self, var, data[var])

            '# 3. 恢复相位分析数据'
            phase_vars = [
                'phase', 'phase_unwrapped', 'phase_compensated',
                'phase_compensation_mat', 'phase_corrected'
            ]
            for var in phase_vars:
                if var in data:
                    setattr(self, var, data[var])

            '# 4. 其他数据'
            other_vars = [
                'window_type', 'window_function'
            ]
            for var in other_vars:
                if var in data:
                    setattr(self, var, data[var])

            '# 5. 恢复定量分析数据'
            analysis_vars = [
                'particle_size', 'particle_num', 'AOP', 'DOLP'
            ]
            for var in analysis_vars:
                if var in data:
                    setattr(self, var, data[var])

            '# 6. 恢复config'
            if 'config' in data:
                self.config = pickle.loads(data['config'])

            msg = f"Data Loaded from {data_url}"
        except Exception as e:
            msg = f"Fail to Load Data: {str(e)}"

        return msg

if __name__ == '__main__':

    test = Hologram()
    # test.save_data()
    # test.load_data()
    print('Hologram is ready.')
