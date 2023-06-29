import json
from Manager.utils import dynamic_loading
from pathlib import Path

def run():
    import sys
    config_file = sys.argv[1]
    print(f'config_file is {config_file}')
    with open(config_file,'r') as f:
        config = json.load(f)
    manger = dynamic_loading(Path(config["Manager"]["path"]),[config["Manager"]["class_name"]])[config["Manager"]["class_name"]](config,save_all_code=True)
    manger.run()