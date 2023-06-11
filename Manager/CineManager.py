import torch
from toolbox.Manager.Manager import BasicManager
from toolbox.Manager.utils import dynamic_loading
from pathlib import Path
from torch.utils.data import DataLoader
import os

class CineManager(BasicManager):
    def __init__(self, config:dict,excute_pipeline:None or list = None,save_all_code:bool = False,load_strict:bool=True):
        super(CineManager,self).__init__(config,save_all_code,load_strict)
        if excute_pipeline is None:
            for func_name in self._get_inline_func():
                self.__getattribute__(func_name)()
        
    def _get_dataloaders(self):
        self.dataset_class = dynamic_loading(Path(self.config["DataSet"]["path"]),
                                             [self.config["DataSet"]["class_name"]])[self.config["DataSet"]["class_name"]]
        self.train_dataset = self.dataset_class(Path(self.config["train"]["DataSet"]["train"]["csv_path"]))
        if "val" in self.config["train"]["DataSet"]:
            self.val_dataset = self.dataset_class(Path(self.config["train"]["DataSet"]["val"]["csv_path"]))
        else:
            self.val_dataset = None
        nw = min([os.cpu_count(), self.config["train"]["batch_size"]["train"] if self.config["train"]["batch_size"]["train"] > 1 else 0, 8])
        self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.config["train"]["batch_size"]["train"],
                                            pin_memory=True,
                                            num_workers=nw)
        if self.val_dataset is not None:
            nw = min([os.cpu_count(), self.config["train"]["batch_size"]["val"] if self.config["train"]["batch_size"]["val"] > 1 else 0, 8])
            self.val_dataloader = DataLoader(self.val_dataset,
                                            batch_size=self.config["train"]["batch_size"]["val"],
                                            pin_memory=True,
                                            num_workers=nw)
    
    def prediction(self):
        """
        心脏电影工作专用的prediction
        """
        from tqdm import tqdm
        from toolbox.Plot.plot import imwrite
        import json
        assert "validate" in self.config
        assert "DataSet" in self.config["validate"]
        assert "csv_path" in self.config["validate"]["DataSet"]
        validate_dataset = self.dataset_class(Path(self.config["validate"]["DataSet"]["csv_path"]))
        validate_dataloader = DataLoader(validate_dataset,shuffle=False)
        draw_path = self.save_path / "figure"
        if not os.path.exists(draw_path):
            os.makedirs(draw_path)
        validate_metrics = {}
        for index, data_items in tqdm(enumerate(validate_dataloader)):
            file_name = Path(validate_dataset.examples[index][0]).stem
            result_dict = self._prediction(data_items)
            undersampling_images = torch.abs(result_dict['model_inputs'][0][0]) # F H W
            recon_images = torch.abs(result_dict['model outputs'][0][0])
            fully_images = result_dict['labels'][0][0]
            for frame_index in range(undersampling_images.shape[0]):
                margin = torch.zeros(undersampling_images[frame_index].shape[0],int(0.1*undersampling_images[frame_index].shape[1])).to(recon_images.device)
                temp_draw_datas = [undersampling_images[frame_index] / undersampling_images[frame_index].max(),
                                   margin,recon_images[frame_index] / recon_images[frame_index].max(),
                                   margin,fully_images[frame_index] / fully_images[frame_index].max()]
                temp_draw_datas = torch.concat(temp_draw_datas, dim=1)
                imwrite(temp_draw_datas,draw_path / (file_name+'_frame_'+str(frame_index+1)+'.png'))
                undersampling_metrics = {}
                recon_metrics = {}
                for metric in self.metrics_dict:
                    recon_metrics[metric] = self.metrics_dict[metric]([fully_images[frame_index].unsqueeze(0).unsqueeze(0)],
                                                                     [recon_images[frame_index].unsqueeze(0).unsqueeze(0)]).item()
                    undersampling_metrics[metric] = self.metrics_dict[metric]([fully_images[frame_index].unsqueeze(0).unsqueeze(0)],
                                                                     [undersampling_images[frame_index].unsqueeze(0).unsqueeze(0)]).item()
                validate_metrics[file_name+'_frame_'+str(frame_index+1)] = {'recon':recon_metrics,
                                                                            'undersampling':undersampling_metrics}
        with open(self.save_path/'validate_metrics.json','w') as f:
            json.dump(validate_metrics,f)

            
            