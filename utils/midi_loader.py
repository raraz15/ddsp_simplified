from typing import List, Dict, Callable

import numpy as np
import pretty_midi
from pretty_midi import Instrument, ControlChange, Note


class MidiLoader:
    FEATURE_VELOCITY = 'velocity'
    FEATURE_PITCH = 'pitch'
    FEATURE_CC_PREFIX = 'cc_'
    FEATURE_DISTANCE_FROM_ONSET = 'distance_from_onset'

    def load(self, midi_file_name: str, frame_rate: int, audio_length_seconds: float):
        midi_data = pretty_midi.PrettyMIDI(midi_file_name)

        instruments: List[Instrument] = midi_data.instruments

        if len(instruments) != 1:
            raise Exception("only midi files with exactly one instrument are supported " +
                            f"{midi_file_name} has {len(instruments)}")

        return self._load_instrument(instruments[0], audio_length_seconds, frame_rate)

    def _load_instrument(self,
                         instrument: Instrument,
                         audio_length_seconds: float,
                         frame_rate: int) -> Dict[str, np.ndarray]:

        time_of_last_midi_event = instrument.get_end_time()

        if time_of_last_midi_event > audio_length_seconds:
            raise Exception(
                f"time of last event in midi: {time_of_last_midi_event} overshoots the end of the audio: {audio_length_seconds}")

        res: Dict[str, np.ndarray] = {}

        control_changes: List[ControlChange] = instrument.control_changes
        control_changes_by_cc_number = self._get_cc_number_changes_dict(control_changes)

        for control_number, control_changes in control_changes_by_cc_number.items():
            control_number: int
            data_for_one_cc = self._generate_data_for_one_cc(audio_length_seconds, frame_rate,
                                                             control_changes_by_cc_number[control_number])
            res[MidiLoader.FEATURE_CC_PREFIX + str(control_number)] = data_for_one_cc

        def get_pitch(note: Note) -> np.float32:
            return np.float32(note.pitch)

        def get_velocity(note: Note) -> np.float32:
            return np.float32(note.velocity)

        res[self.FEATURE_VELOCITY] = self._generate_single_value_data_for_notes(instrument=instrument,
                                                                                frame_rate=frame_rate,
                                                                                audio_length_seconds=audio_length_seconds,
                                                                                get_single_feature_value_from_note_func=get_velocity)

        res[self.FEATURE_PITCH] = self._generate_single_value_data_for_notes(instrument=instrument,
                                                                             frame_rate=frame_rate,
                                                                             audio_length_seconds=audio_length_seconds,
                                                                             get_single_feature_value_from_note_func=get_pitch)

        res[self.FEATURE_DISTANCE_FROM_ONSET] = self._generate_distance_from_onset_data(
            instrument,
            frame_rate,
            audio_length_seconds
        )

        return res

    def _get_cc_number_changes_dict(self, control_changes: List[ControlChange]) -> Dict[int, List[ControlChange]]:
        control_changes_by_cc_number: Dict[int, List[ControlChange]] = {}
        for control_change in control_changes:
            if control_change.number not in control_changes_by_cc_number:
                control_changes_by_cc_number[control_change.number] = []
            control_changes_by_cc_number[control_change.number].append(control_change)

        return control_changes_by_cc_number

    def _generate_data_for_one_cc(self,
                                  audio_length_seconds: float,
                                  frame_rate: int,
                                  control_changes: List[ControlChange]) -> np.ndarray:
        """
        Generate a numpy array with CC values, assume that all ControlChanges are related to the same CC.
        """
        num_frames = int(audio_length_seconds * frame_rate)
        res = np.zeros(dtype=np.float32, shape=(num_frames,))

        cc_number = control_changes[0].number

        # adding a virtual 0 change at the very beginning
        virtual_initial_cc_change = ControlChange(number=cc_number, value=0, time=0)

        augmented_changes = [virtual_initial_cc_change, *control_changes]

        for i, change in enumerate(augmented_changes):
            time_of_this_change = change.time
            time_of_next_change: float
            if i < len(augmented_changes) - 1:
                time_of_next_change = augmented_changes[i + 1].time
            else:
                time_of_next_change = audio_length_seconds

            start_idx = int(time_of_this_change * frame_rate)
            end_idx = int(time_of_next_change * frame_rate)

            res[start_idx:end_idx] = np.float32(change.value)

        return res

    def _generate_single_value_data_for_notes(self, instrument: Instrument, frame_rate: int,
                                              audio_length_seconds: float,
                                              get_single_feature_value_from_note_func: Callable[
                                                  [Note], np.float32]) -> np.ndarray:
        """
        Get arbitrary note-bound single value feature data.

        In fact, it's just about having one function for both velocity and pitch.
        """

        num_frames = int(audio_length_seconds * frame_rate)
        res = np.zeros(dtype=np.float32, shape=(num_frames,))

        for note in instrument.notes:
            note: Note

            start_idx = int(note.start * frame_rate)
            end_idx = int(note.end * frame_rate)
            res[start_idx:end_idx] = get_single_feature_value_from_note_func(note)

        return res

    def _generate_distance_from_onset_data(self, instrument: Instrument, frame_rate: int, audio_length_seconds: float):
        """
        Generate data of "distance from onset" feature
        """
        num_frames = int(audio_length_seconds * frame_rate)
        res = np.zeros(dtype=np.float32, shape=(num_frames,))

        for note in instrument.notes:
            note: Note

            start_idx = int(note.start * frame_rate)
            end_idx = int(note.end * frame_rate)

            final_value = note.end - note.start
            patch = np.linspace(0, final_value, num=end_idx - start_idx, dtype=np.float32)
            res[start_idx:end_idx] = patch

        return res
