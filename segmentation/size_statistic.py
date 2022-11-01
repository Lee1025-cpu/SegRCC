# _*_ coding:utf-8 _*_
# Lee 2022-05-09 10:56 size_statistic
# Note: 

import argparse
from Now_state.segmentation.utils import *


def main(config):
    if not os.path.exists(config.out_saving_path):
        os.makedirs(config.out_saving_path)

    # # spacing&size统计：spcaing选取[0.6, 0.6, 0.6] size选取
    # spacing_size_statistic(config)
    #
    # # 像素重采样至选取spacing
    # respacing_data(config)
    #
    # # size重统计 选取[192, 192, 256]
    size_statistic(config)

    # 像素值归一化
    # 0503: pixel value看起来像是归一化过的 所以未进行归一化 [0, 3135]
    # pixel_value_statistic(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    root = "D:/Dataset/1/debug_data_saving/"

    parser.add_argument("--root", default=root)
    parser.add_argument("--image_path", default=root + "f4_d0.4A0.1_80_0.05_rre_0.0001_S_s10_d0.8_dc_mean_re_phase2_/0/")
    parser.add_argument("--label_path", default=root + "f4_d0.4A0.1_80_0.05_rre_0.0001_S_s10_d0.8_dc_mean_re_phase2_/0/")
    parser.add_argument("--out_saving_path", default=root + "f4_d0.4A0.1_80_0.05_rre_0.0001_S_s10_d0.8_dc_mean_re_phase2_/")

    config = parser.parse_args()
    main(config)
