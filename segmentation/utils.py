# _*_ coding:utf-8 _*_
# Lee 2022-05-09 10:57 utils
# Note: 

import os
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import plotly
import plotly.graph_objs as go


def size_statistic(config):
    Img_list = os.listdir(config.image_path)
    img_list = []
    for item in Img_list:
        if "lab" in item:
            img_list.append(item)

    out_statistic_data = []
    excel_saving_path = config.out_saving_path + 'new_size_statistic.xlsx'

    if not os.path.exists(excel_saving_path):
        for index_ in tqdm(range(len(img_list))):
            img_path = config.image_path + img_list[index_]
            img = sitk.ReadImage(img_path)
            size = img.GetSize()
            out_statistic_data.append([size[0], size[1], size[2]])

        data = pd.DataFrame(data=out_statistic_data, index=img_list, columns=["Size0", "Size1", "Size2"])
        data.to_excel(excel_saving_path)

    data1 = pd.read_excel(excel_saving_path, sheet_name='Sheet1')
    if not os.path.exists(config.out_saving_path + "new_3D_data_size.html"):
        fig2 = go.Scatter3d(x=data1['Size0'], y=data1['Size1'], z=data1['Size2'],
                            marker=dict(opacity=0.9,
                                        reversescale=True,
                                        colorscale='Blues',
                                        size=5),
                            line=dict(width=0.02),
                            mode='markers')

        mylayout2 = go.Layout(scene=dict(xaxis=dict(title="Size0"),
                                         yaxis=dict(title="Size1"),
                                         zaxis=dict(title="Size2")), )

        plotly.offline.plot({"data": [fig2], "layout": mylayout2}, auto_open=True, filename=(config.out_saving_path +
                                                                                             "new_3D_data_size.html"))


if __name__ == "__main__":
    pass

