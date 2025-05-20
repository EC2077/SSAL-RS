SSAL-高分辨率遥感图像分类
本项目基于自监督自生成学习（Self-Supervised Autogenous Learning, SSAL）方法，首次将其应用于高分辨率遥感图像分类任务，并在 UCMerced_LandUse 和 RSI-CB128 数据集上进行了实验评估。该方法可显著减少对人工标注数据的依赖，并提升模型的分类精度与泛化能力。

请使用以下 Python 环境运行本项目：
python >= 3.8
tensorflow-gpu == 2.9.0

请按以下方式准备数据集文件夹结构：
datasets/
├── UCMerced_LandUse/
│   ├── airplane/
│   ├── forest/
│   └── ...
└── RSI-CB128/
    ├── airport_runway/
    ├── desert/
    └── ...

运行方式：调整好参数后，在终端中执行python save_output.py命令




