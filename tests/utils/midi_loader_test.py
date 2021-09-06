import pathlib
import unittest

from utils.midi_loader import MidiLoader


class MyTestCase(unittest.TestCase):
    def test_load_file(self):
        loader = MidiLoader()

        test_midi_file_path = str(pathlib.Path(__file__).parent.resolve()) + '/data/test.MID'
        res = loader.load(test_midi_file_path, 250, 4)

        keys = res.keys()
        pass
