import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import FunctionType
from typing import List
from pathlib import Path
from importlib import import_module
import sys

def get_now_datetime():
    """
    @Description: 返回当前时间，格式为：年月日时分秒
    """
    import time
    return time.strftime('%Y年%m月%d日%H点%M分', time.localtime(time.time()))[2:]


def dynamic_loading(filepath:Path, contents:List):
    """
    动态从python文件加载特定的实例
    :param filepath: config文件里的python文件路径
    :param contents: 待加载的python文件的实例
    """
    parent_path = str(filepath.parent)
    file_name = filepath.stem
    result = {}
    sys.path.append(str(parent_path))
    module_ = import_module(file_name)
    for content in contents:
        result[content] = eval(f'module_.{content}')  # drity import
    sys.path.remove(str(parent_path))
    return result


def inference(model:torch.nn.Module, optimizer:torch.optim.Optimizer or None,
              data_loader:DataLoader, epoch:int, loss_func:FunctionType,pretreatment_func:FunctionType,
              metrics_dict = {}, mode='train',ncols=100):
    """
    对模型进行推理
    :param model: 用于推理的模型
    :param optimizer: 用于梯度更新参数的优化器
    :param data_loader: 数据加载器
    :param epoch: 迭代次数
    :param loss_func: 损失函数 loss_func(gts,model_outputs) -> loss_value
    :param pretreatment_func: 预处理函数 pretreatment_func(gts:list, model_inputs:list) -> gts:list, model_inputs:list
    :param metrics_dict:用于计算推理中需要记录的指标, eg. {'PSNR':cal_psnr(gts,model_outputs)}
    :param mode:用于判定该推理是否在训练流程中, 从而启动dropout等类型的网络层4
    :param ncols: 用于指定tqdm进度条的长度
    """
    if mode != 'train':
        model.eval()
    else:
        model.train()
    device = list(model.parameters())[0].device
    metrics = {'Loss':torch.zeros(1).to(device)}
    for extra_metric in metrics_dict:
        metrics[extra_metric] = torch.zeros(1).to(device)
    if optimizer is not None:
        optimizer.zero_grad()
    batch = torch.tensor(0, dtype=torch.int).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=ncols)
    for data_items in data_loader:
        data_items = [item.to(device) for item in data_items]
        gts,model_inputs = pretreatment_func(data_items)
        model_outputs = model(model_inputs)
        # 计算指标
        with torch.no_grad():
            for extra_metric in metrics_dict:
                temp_metric = metrics_dict[extra_metric](gts,model_outputs)
                metrics[extra_metric] = (metrics[extra_metric] * batch + temp_metric.detach()) / (batch + 1)
        # 计算Loss
        loss = loss_func(gts,model_outputs)
        metrics['Loss'] = (metrics['Loss'] * batch + loss.detach()) / (batch + 1)
        data_loader.desc = f"[{mode} epoch {epoch}]" + '\t'.join([metric+': '+str(round(metrics[metric].item(),5)) for metric in metrics])
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch += 1
    return {k:metrics[k].item() for k in metrics}

def trace_all_dict_from_dict(D:dict):
    """
    获取一个字典中所有层级的字典
    :param D: 源字典
    """
    result = [D]
    trace_queue = [D]
    while len(trace_queue) > 0:
        temp_D = trace_queue.pop(0)
        for key in temp_D:
            if isinstance(temp_D[key],dict):
                trace_queue.append(temp_D[key])
                result.append(temp_D[key])
    return result
