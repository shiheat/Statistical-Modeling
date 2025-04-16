import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# 确保文件夹存在
output_dir = r"E:\学习\统计建模\5楼\outs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 导入数据
file_path = r"E:\学习\统计建模\5楼\20230912data5.csv"  # 本地CSV文件路径
data = pd.read_csv(file_path, encoding='GBK')

# 确保数据按时间排序
data['ts'] = pd.to_datetime(data['ts'])
data = data.sort_values(by='ts')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或使用 'SimHei'，确保matplotlib支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为乱码的问题

# 定义绘图函数
def plot_visualizations(region_data):
    # 1. 交互式时间序列图
    fig1 = px.line(region_data, x="ts", y=["电压", "电流", "色温", "亮度"], title=f"{region_data['region'].iloc[0]} - 时间序列图")
    fig1.update_layout(
        xaxis_title="时间",
        yaxis_title="传感器数值",
        template="plotly_dark"
    )
    fig1.show()
    fig1.write_image(os.path.join(output_dir, f"{region_data['region'].iloc[0]}_时间序列图.png"))

    # 2. 3D散点图
    fig2 = px.scatter_3d(region_data, x="电压", y="电流", z="温度", color="region", title=f"{region_data['region'].iloc[0]} - 3D散点图")
    fig2.update_layout(
        scene = {
            'xaxis_title': '电压',
            'yaxis_title': '电流',
            'zaxis_title': '温度'
        },
        template="plotly_dark"
    )
    fig2.show()
    fig2.write_image(os.path.join(output_dir, f"{region_data['region'].iloc[0]}_3D散点图.png"))

    # 3. 桑基图
    labels = ["无人", "有人", "关", "开"]
    sources = [0, 1, 1, 2]  # 示例数据流动源
    targets = [2, 3, 0, 3]  # 示例数据流动目标
    values = [10, 20, 30, 40]  # 示例数据流的大小

    fig3 = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig3.update_layout(title=f"{region_data['region'].iloc[0]} - 桑基图", template="plotly_dark")
    fig3.show()
    fig3.write_image(os.path.join(output_dir, f"{region_data['region'].iloc[0]}_桑基图.png"))

    # 4. 密度图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(region_data['电压'], shade=True, color="r", label="电压")
    sns.kdeplot(region_data['电流'], shade=True, color="g", label="电流")
    sns.kdeplot(region_data['色温'], shade=True, color="b", label="色温")
    plt.title(f"{region_data['region'].iloc[0]} - 传感器数据分布")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{region_data['region'].iloc[0]}_传感器数据分布.png"))
    plt.close()

# 根据region生成不同的图表
regions = data['region'].unique()  # 获取所有唯一的区域

for region in regions:
    region_data = data[data['region'] == region]  # 获取每个区域的数据
    plot_visualizations(region_data)

print(f"图像已保存到：{output_dir}")
