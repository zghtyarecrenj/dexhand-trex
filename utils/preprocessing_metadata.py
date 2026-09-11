"""Preserve collection RGB preprocessing with checkpoints; images are already processed.

No dependency on the collection repository or its camera/OpenCV modules.
"""
import json
from pathlib import Path


def capture_preprocessing(args):
    if getattr(args,'data_format','json')=='lerobot':
        directory=Path(args.lerobot_root)/'meta'
    else:
        directory=Path(args.data_path).parent
    path=directory/'image_preprocessing.json'
    if not path.is_file():
        return None  # Legacy datasets: no custom crop metadata was provided.
    config=json.loads(path.read_text())
    if not isinstance(config,dict) or config.get('schema')!='collection_camera_crop' or type(config.get('version')) is not int or config['version']!=1:
        raise ValueError(f'Unsupported image preprocessing metadata: {path}')
    if config.get('coordinate_space')!='raw_camera_pixels' or config.get('resize')!='letterbox_inter_area' or not isinstance(config.get('cameras'),dict):
        raise ValueError(f'Invalid image preprocessing metadata: {path}')
    return config


def write_preprocessing(directory,config):
    if config is not None:
        (Path(directory)/'image_preprocessing.json').write_text(json.dumps(config,indent=2)+'\n')
