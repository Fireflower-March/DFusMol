# DFusMol 代码位置速览

## 论文模块对应

- 基于基序的结构表示：`preprocess/`、`dataset/data.py`
- 分子图的构建：`featureset/`、`preprocess/`
- 图编码器：`modelset/cmpn.py`
- 双通道特征融合：`modelset/lgam.py`
- 全局-局部注意力机制：`modelset/model_atom_level_pytorch.py`、`modelset/model_motif_level_pytorch.py`

## 文件夹速览

`modelset/` 文件夹为核心模型功能：

- `model.py` 文件为整体模型组装功能；
- `cmpn.py` 文件为分子图编码器功能，是图编码器核心；
- `lgam.py` 文件为双通道特征融合功能，是融合模块核心；
- `model_atom_level_pytorch.py` 文件为原子级通道与全局-局部注意力功能；
- `model_motif_level_pytorch.py` 文件为基序级通道与全局-局部注意力功能；
- `__init__.py` 文件为模块导出功能。

`preprocess/` 文件夹为离线预处理功能：

- `preprocessing.py` 文件为预处理入口功能，用来生成预处理后的 `.npy` 文件；
- `utils.py` 文件为基序分解与 token 化功能，是基序表示核心；
- `dataset_processed.py` 文件为邻接矩阵、距离矩阵、原子-基序匹配矩阵构建功能；
- `mol_graph.py` 文件为原子层图结构整理功能；
- `token_id.json` 文件为基序词表功能。

`featureset/` 文件夹为图特征构建功能：

- `atom_featurization.py` 文件为原子特征与键特征构建功能；
- `tograph.py` 文件为把 SMILES 转成分子图输入功能，是 `CMPN` 前面的关键文件；
- `motif_featurization.py` 文件为辅助特征封装功能，主流程中不算最核心。

`dataset/` 文件夹为数据读取与 batch 组织功能：

- `data.py` 文件为样本封装功能，是数据输入核心；
- `batchset.py` 文件为 batch 对齐与 padding 功能；
- `utils.py` 文件为读入 CSV、构建数据集、划分训练验证测试集功能；
- `scaffold.py` 文件为 scaffold split 和 cluster split 功能；
- `scaler.py` 文件为回归标签标准化功能；
- `__init__.py` 文件为数据模块导出功能。

`trainset/` 文件夹为训练与评估功能：

- `parsing.py` 文件为训练参数定义功能；
- `a_run_train.py` 文件为单次完整训练流程功能，是训练主流程核心；
- `training.py` 文件为每个 batch / epoch 的训练功能；
- `predict.py` 文件为模型预测功能；
- `evaluate.py` 文件为评估指标计算功能；
- `utils.py` 文件为优化器、学习率、checkpoint 等通用功能；
- `nn_utils.py` 文件为神经网络辅助功能；
- `logger.py` 文件为日志与实验目录管理功能；
- `__init__.py` 文件为训练模块导出功能。

## 最建议你讲的主线

如果老师问“代码主要在哪”，按下面顺序讲最清楚：

1. `train.py`：训练入口；
2. `dataset/`：数据怎么进模型；
3. `preprocess/`：基序怎么拆；
4. `featureset/`：分子图和特征怎么建；
5. `modelset/cmpn.py`：图编码器；
6. `modelset/lgam.py`：双通道融合；
7. `modelset/model_atom_level_pytorch.py`、`modelset/model_motif_level_pytorch.py`：注意力机制；
8. `trainset/`：训练、预测、评估。
