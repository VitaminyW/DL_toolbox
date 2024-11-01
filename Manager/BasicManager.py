import torch
import json
from pathlib import Path,PosixPath
from Manager.utils import inference, dynamic_loading, get_now_datetime, trace_all_dict_from_dict
import os
import shutil
from torch.utils.tensorboard import SummaryWriter # type: ignore
import warnings
from Monitor.reminder import get_reminder
import time
from typing import List,Dict,Union

class BasicManager:
    """流程控制的基本类, 其基础类方法如下
        _get_device: 设置训练所使用的设备(如果存在多个GPU设备, 自动使用数据并行)
        _get_model: 从字典"Model"字段设置中加载模型
        _get_optim: 从字典"Optimizer"字段设置中加载优化器设置
        _get_dataloaders: 获取dataloader设置
        _model_data_parallel: 如果存在多个GPU设备, 自动使用数据并行
        _get_super_params: 获取字典"train"字段超参数(epoch,time_per_val,time_per_save)
        _setting_logger_and_save_path: 从字典"Other"字段加载保存的目标路径、Bart的token以及tensorboard的文件名
        _get_loss_func: 从字典"train"字段中的"Loss"字段加载损失函数
        _get_metrics_dict:从字典"train"字段中的"Metrics_Dict"字段加载评估函数字典
        _get_preprocessing:从字典"train"字段中的"Preprocessing"字段加载预处理函数
    """
    def __init__(self, config:dict, save_all_code:bool = False, load_strict:bool = True, excute_pipeline:Union[bool, None] = None,**argv):
        """
        Args:
            config (dict): 读取的json配置信息
            save_all_code (bool, optional): 设置是否将所遇到的代码文件进行保存. Defaults to False.
            load_strict (bool, optional): 加载的参数是否需要一一对应. Defaults to True.
            excute_pipeline (Union[bool, None], optional): 内部函数的执行过程,详情看_inline_func. Defaults to None.
        """
        self.config:dict  = config
        self.save_all_code:bool = save_all_code
        self.load_strict:bool = load_strict
        self._check_basic_config()
        if excute_pipeline is None:
            for func_name in self._inline_func:
                self.__getattribute__(func_name)()
    
    @property
    def _inline_func(self)->List[str]:
        """返回基础类方法名

        Returns:
            _type_: _description_
        """
        return ['_get_device','_get_model','_get_optim','_get_dataloaders',
                '_model_data_parallel','_get_super_params','_setting_logger_and_save_path',
                '_get_loss_func','_get_metrics_dict','_get_preprocessing']
        

    def _check_basic_config(self):
        assert "train" in self.config, "配置文件中需要存在train字段"
        if "device" not in self.config["train"] or len(self.config["train"]["device"])==0:
            self.config["train"] = ["cpu"]
        assert "Model" in self.config,"配置文件中需要存在Model字段"
        assert "path" in  self.config["Model"], "Model字段中需要明确模型文件的地址"
        assert "class_name" in  self.config["Model"], "Model字段中需要明确模型的类名"
        assert "parameters" in  self.config["Model"], "Model字段中需要明确模型的参数,可以是空的字典"
        assert "DataSet" in self.config, "配置文件中需要存在DataSet字段"
        assert "path" in  self.config["DataSet"], "DataSet字段中需要明确Dataset文件的地址"
        assert "class_name" in  self.config["DataSet"], "DataSet字段中需要明确模型的类名"
        if "epoch" not in self.config["train"]:
            self.config["train"]["epoch"] = 100 # 默认迭代次数为100
        if "batch_size" not in self.config["train"]:
            self.config["train"]["batch_size"] = 7 # 默认mini-batch size 为7
        if isinstance(self.config["train"],int):
            self.config["train"]["batch_size"] = {'train':self.config["train"]["batch_size"],
                                                  'val':self.config["train"]["batch_size"]}
        if "checkpoint_path" not in self.config["train"]:
            self.config["train"]["checkpoint_path"] = ''
        if "times_per_valid" not in self.config["train"]:
            self.config["train"]["times_per_valid"] = 10# 默认times_per_valid 为10
        if "times_per_save" not in self.config["train"]:
            self.config["train"]["times_per_save"] = self.config["train"]["times_per_valid"]# 默认times_per_save 为times_per_valid
        assert "Optimizer" in self.config, "配置文件中需要存在Optimizer字段"
        if "Name" not in self.config["Optimizer"]:
            self.config["Optimizer"]["Name"] = "Adam" # 默认优化器为Adam
        if "lr" not in self.config["Optimizer"]:
            self.config["Optimizer"]["lr"] = 1e-3 # 默认学习率为1e-3
        if  "Other" not in self.config:
            self.config["Other"] = {"save_path":"./exprs/",
                                                  "reminder_token":"",
                                                  "tb_writer":"logs"}
        if "save_path" not in self.config["Other"]:
            self.config["Other"]["save_path"]="./exprs/"
        if "reminder_token" not in self.config["Other"]:
            self.config["Other"]["reminder_token"] = ""
        if "tb_writer" not in self.config["Other"]:
            self.config["Other"]["tb_writer"] = "logs"
        
        
    def _get_device(self):
        """ 设置训练所使用的设备
        """
        self.device = self.config["train"]["device"]
        if not isinstance(self.device,list):
            self.device = [self.device]

        
    def _get_model(self, other_condition_model_params:Union[dict,None] = None):
        """从字典"Model"字段设置中加载模型

        Args:
            other_condition_model_params (Union[dict,None], optional): 如果模型需要传入类实例或其他无法用json表达的属性时,
                                                                        可以利用该参数进行传参. Defaults to None.
        """
        
        if other_condition_model_params is None:
            self.model = dynamic_loading(Path(self.config["Model"]["path"]),[self.config["Model"]["class_name"]])[self.config["Model"]["class_name"]](**self.config["Model"]["parameters"])
        else:
            self.model = dynamic_loading(Path(self.config["Model"]["path"]),[self.config["Model"]["class_name"]])[self.config["Model"]["class_name"]](**other_condition_model_params)
        self.model = self.model.to(self.device[0])
        # 加载参数
        if len(self.config["train"]["checkpoint_path"]) > 0 and os.path.exists(self.config["train"]["checkpoint_path"]):
            print(f'loading checkpoint_path from {self.config["train"]["checkpoint_path"]}')
            weights_dict = torch.load(self.config["train"]["checkpoint_path"], map_location=self.device[0])
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                if self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(load_weights_dict, strict=self.load_strict)
    
    def _get_optim(self):
        """从字典"Optimizer"字段设置中加载优化器设置
        """
        pg = [p for p in self.model.parameters() if p.requires_grad]
        optim_para = {'lr':self.config['Optimizer']['lr'],"params":pg}
        self.optim = eval(f"torch.optim.{self.config['Optimizer']['Name']}(**optim_para)")

    def _model_data_parallel(self):
        """如果存在多个GPU设备, 自动使用数据并行
        """
        if len(self.device) > 1 and 'cpu' not in self.device:
            device_ids = []
            for item in self.device:
                t = item.replace(' ','').split(':')
                if len(t) == 2:
                    device_ids.append(int(t[1]))
                else:
                    device_ids.append(0)
            self.model = torch.nn.DataParallel(self.model,device_ids=device_ids).to(self.device[0])
    
    def _get_super_params(self):
        """ 获取字典"train"字段超参数(epoch,time_per_val,time_per_save)
        """
        self.epochs = self.config["train"]["epoch"]
        self.times_per_valid = self.config["train"]["times_per_valid"]
        self.times_per_save = self.config["train"]["times_per_save"]
    
    def _setting_logger_and_save_path(self):
        """从字典"Other"字段加载保存的目标路径、Bart的token以及tensorboard的文件名
        """
        self.save_path = self.config["Other"]["save_path"]
        now_time = get_now_datetime()
        self.save_path = Path(self.save_path+ Path(self.config["Model"]["path"]).stem + '_' +self.config["Model"]["class_name"] + now_time)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            # 避免在同一分钟运行多次
            self.save_path = Path(str(self.save_path)+time.strftime('%S秒', time.localtime(time.time())))
            os.makedirs(self.save_path)
        if not os.path.exists(self.save_path / 'params'):
            os.makedirs(self.save_path / 'params')
        # 记录相关类
        self.tb_writer = SummaryWriter(str(self.save_path / self.config["Other"]["tb_writer"]))
        if len(self.config["Other"]["reminder_token"]) > 5:
            self.train_remider = get_reminder(self.config["Other"]["reminder_token"],self.times_per_valid,now_time)
            self.val_remider = get_reminder(self.config["Other"]["reminder_token"],1,now_time)
        else:
            self.train_remider = None
            self.val_remider = None
        # 将配置信息保存至当前实验的目录下
        with open(self.save_path / 'config.json','w') as f:
                json.dump(self.config,f)
        if self.save_all_code:
            # 保存相关代码文件
            code_save_path = self.save_path / 'codes'
            if not os.path.exists(code_save_path):
                os.makedirs(code_save_path)
            dicts = trace_all_dict_from_dict(self.config)
            code_paths = [Path(item['path']) for item in dicts if 'path' in item]
            for code_path in code_paths:
                shutil.copy(code_path, code_save_path / code_path.name)
    
    def _get_loss_func(self):
        """从字典"train"字段中的"Loss"字段加载损失函数
        """
        self.loss_func = dynamic_loading(Path(self.config["train"]["Loss"]["path"]),[self.config["train"]["Loss"]["name"]])[self.config["train"]["Loss"]["name"]]
    
    def _get_metrics_dict(self):
        """从字典"train"字段中的"Metrics_Dict"字段加载评估函数字典
        字典形如: {'Accuracy':cal_accuracy(Callable),'R1':cal_r1(Callabel)}
        """
        if "Metrics_Dict" in self.config["train"]:
            self.metrics_dict = dynamic_loading(Path(self.config["train"]["Metrics_Dict"]["path"]),[self.config["train"]["Metrics_Dict"]["name"]])[self.config["train"]["Metrics_Dict"]["name"]]
        else:
            self.metrics_dict = {}
            
    def _get_preprocessing(self):
        """从字典"train"字段中的"Preprocessing"字段加载预处理函数
        """
        if "Preprocessing" in self.config["train"]:
            self.preprocessing_func = dynamic_loading(Path(self.config["train"]["Preprocessing"]["path"]),[self.config["train"]["Preprocessing"]["name"]])[self.config["train"]["Preprocessing"]["name"]]
        else:
            self.preprocessing_func = lambda item:item
            
    def run(self):
        """执行训练流程
        """
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %('
                                                'message)s')
        warnings.filterwarnings('ignore')
        for i in range(self.epochs):
            train_epoch_metrics = inference(self.model,self.optim,self.train_dataloader,i,self.loss_func,self.preprocessing_func,self.metrics_dict,'train')
            for metric in train_epoch_metrics:
                self.tb_writer.add_scalar(tag=f'Training_{metric}', scalar_value=train_epoch_metrics[metric],global_step=i)
            if (i+1)%self.times_per_valid == 0:
                val_epoch_metrics = inference(self.model,self.optim,self.val_dataloader,i,self.loss_func,self.preprocessing_func,self.metrics_dict,'val')
                for metric in val_epoch_metrics:
                    self.tb_writer.add_scalar(tag=f'Validation_{metric}', scalar_value=val_epoch_metrics[metric],global_step=i)
                if self.val_remider is not None:
                    self.val_remider(val_epoch_metrics,f'Epoch-{i}-val')
            if self.train_remider is not None:
                self.train_remider(train_epoch_metrics,f'Epoch-{i}-train')
            if (i+1)%self.times_per_save == 0:
                torch.save(self.model.state_dict(), self.save_path / 'params' / f"model-{i}.pth")
    
    @torch.no_grad()
    def _prediction(self, validate_inputs:list):
        """
        使用模型预测一批数据
        :param validate_inputs: 包含模型输入的inputs列表
        return {"model_inputs":model_inputs,"labels":gts,"model outputs":outputs}
        """
        device = list(self.model.parameters())[0].device
        validate_inputs = [item.to(device) for item in validate_inputs]
        gts,model_inputs = self.preprocessing_func(validate_inputs)
        self.model.eval()
        outputs = self.model(model_inputs)
        return {"model_inputs":model_inputs,"labels":gts,"model_outputs":outputs}
    
    def prediction(self):
        results = []
        for data_items in self.val_dataloader:
            results.append(self._prediction(data_items))
        return results
    
    def _get_dataloaders(self):
        self.train_dataloader = None
        self.val_dataloader = None
        raise NotImplementedError('该类为基本类, 需要子类实现该方法')