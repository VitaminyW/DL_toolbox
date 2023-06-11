import json
from toolbox.Manager.utils import dynamic_loading
from pathlib import Path

if __name__ == '__main__':
    import sys
    config_file = sys.argv[1]
    # config_file = '/data/disk1/yewei/MICCA_cine/Code/script/Unet_3d.json'
    print(f'config_file is {config_file}')
    with open(config_file,'r') as f:
        config = json.load(f)
    manger = dynamic_loading(Path(config["Manager"]["path"]),[config["Manager"]["class_name"]])[config["Manager"]["class_name"]](config,save_all_code=True)
    manger.run()