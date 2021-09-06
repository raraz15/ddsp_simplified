from typing import List, Dict

import numpy as np
import pretty_midi
from pretty_midi import Instrument

MIDI_FEATURE_VELOCITY = 'velocity'
MIDI_FEATURE_PITCH = 'pitch'
MIDI_FEATURE_CC_PREFIX = 'cc_'

class MidiLoader:
    def load(self, midi_file_name: str, frame_rate: int, feature_names: List[str]):
        midi_data = pretty_midi.PrettyMIDI(midi_file_name)

        instruments: List[Instrument] = midi_data.instruments

        if len(instruments) != 1:
            raise Exception("only midi files with exactly one instrument are supported " +
                            f"{midi_file_name} has {len(instruments)}")

        return self._load_instrument(instruments[0])


    def _load_instrument(self, instrument: Instrument) -> Dict[str, np.ndarray]:
        pass


