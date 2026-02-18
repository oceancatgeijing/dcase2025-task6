from typing import List, Dict, Any

import torch
from aac_datasets.utils.collections import list_dict_to_dict_list  # 改用官方版本
from torch import Tensor

# def list_dict_to_dict_list(list_dict):
#     """
#     手动实现该工具函数，绕过 aac_datasets 的版本兼容性问题。
#     功能：将 [{key: val1}, {key: val2}] 转换为 {key: [val1, val2]}
#     """
#     if not list_dict:
#         return {}
#     return {key: [d[key] for d in list_dict] for key in list_dict[0].keys()}



class CustomCollate:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dic: Dict[str, Any] = list_dict_to_dict_list(batch_lst)
        keys = list(batch_dic.keys())

        for key in keys:
            values = batch_dic[key]

            if len(values) == 0:
                batch_dic[key] = values
                continue

            are_tensors = [isinstance(value, Tensor) for value in values]
            if not all(are_tensors):
                batch_dic[key] = values
                continue

            if can_be_stacked(values):
                values = torch.stack(values)
                batch_dic[key] = values
                continue

            batch_dic[key] = values
        return batch_dic


def can_be_stacked(tensors: List[Tensor]) -> bool:
    """Returns true if a list of tensors can be stacked with torch.stack function."""
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    are_stackables = [tensor.shape == shape0 for tensor in tensors]
    return all(are_stackables)
