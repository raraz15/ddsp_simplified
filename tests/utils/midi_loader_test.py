import pathlib
import unittest

from utils.midi_loader import MidiLoader


class MyTestCase(unittest.TestCase):
    def test_load_file(self):
        loader = MidiLoader()

        test_midi_file_path = str(pathlib.Path(__file__).parent.resolve()) + '/data/test.MID'
        res = loader.load(test_midi_file_path, 250, 4)

        keys = res.keys()

        self.assertEquals(['cc_55', 'velocity', 'pitch'], list(keys))
        self.assertEqual(1000, res['cc_55'].shape[0])
        self.assertEqual(1000, res['velocity'].shape[0])
        self.assertEqual(1000, res['pitch'].shape[0])

        self.assertEquals(0, min(res['velocity']))
        self.assertEquals(0, min(res['pitch']))
