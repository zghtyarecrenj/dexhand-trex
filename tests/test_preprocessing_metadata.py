import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.preprocessing_metadata import capture_preprocessing,write_preprocessing


class PreprocessingTests(unittest.TestCase):
    def test_json_and_lerobot_snapshot_survives_source_edits(self):
        config=dict(schema='collection_camera_crop',version=1,coordinate_space='raw_camera_pixels',
                    resize='letterbox_inter_area',cameras={'camera:wrist_right':{
                        'source_size':[640,480],'roi':[100,80,400,300]}})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meta').mkdir();(root/'checkpoint').mkdir()
            for kind,directory in (('json',root),('lerobot',root/'meta')):
                args=SimpleNamespace(data_format=kind,data_path=str(root/'train.json'),lerobot_root=str(root))
                path=directory/'image_preprocessing.json'
                self.assertIsNone(capture_preprocessing(args))
                path.write_text(json.dumps(config));snapshot=capture_preprocessing(args)
                path.write_text('{}')
                write_preprocessing(root/'checkpoint',snapshot)
                self.assertEqual(json.loads((root/'checkpoint/image_preprocessing.json').read_text()),config)
                with self.assertRaises(ValueError):capture_preprocessing(args)


if __name__=='__main__':unittest.main()
