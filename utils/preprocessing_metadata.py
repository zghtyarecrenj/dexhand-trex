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
    if not isinstance(config,dict) or config.get('schema')!='collection_camera_crop' or type(config.get('version')) is not int or config['version'] not in (1,2):
        raise ValueError(f'Unsupported image preprocessing metadata: {path}')
    if config.get('coordinate_space')!='raw_camera_pixels' or config.get('resize')!='letterbox_inter_area' or not isinstance(config.get('cameras'),dict):
        raise ValueError(f'Invalid image preprocessing metadata: {path}')
    if config['version']==2:
        sizes=config.get('output_sizes')
        if not isinstance(sizes,dict) or set(sizes)!={'camera:realsense_color','camera:wrist_right'}:
            raise ValueError('Expected exported output sizes for both camera views')
        for size in sizes.values():
            if not isinstance(size,list) or len(size)!=2 or any(type(n) is not int or n<1 for n in size):
                raise ValueError('Image output size must be [positive width,positive height]')
        if getattr(args,'image_size',None) is not None and any(list(args.image_size)!=size for size in sizes.values()):
            raise ValueError('--image_size would resize exported camera views differently from the saved config. '
                             'Omit it to preserve each camera size, or match the dataset output sizes.')
    return config


def write_preprocessing(directory,config):
    if config is not None:
        (Path(directory)/'image_preprocessing.json').write_text(json.dumps(config,indent=2)+'\n')
