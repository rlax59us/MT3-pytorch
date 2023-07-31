from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from data.utils import encode_midi, _snote2events, _make_time_sift_events
import librosa
import numpy as np 

from itertools import cycle
import json
import os

import random
from data.constants import *
from data.mel import *
from config.data_config import data_config

class MIDIDataset(IterableDataset):
    def __init__(self, root_dir='/home/esteban/MAESTRO_dataset', type='train'): 
        # type : train, validation, test
        self.type = type
        self.metadata = json.load(open(os.path.join(root_dir, 'maestro-v3.0.0.json')))
        self.paths = list(Path(root_dir).glob('**/*.mid')) + list(Path(root_dir).glob('**/*.midi'))
        self.midi_paths = list()
        for idx in range(len(self.paths)):
            if self.metadata['split'][str(idx)] == type:
                self.midi_paths.append(self.paths[idx])
            else:
                continue
        self.wav_paths = list()
        for path in self.midi_paths:
            self.wav_paths.append(str(path).replace('.midi', '.wav'))
        self.config = data_config

    def _load_audio(self, audio_path, sample_rate):
        audio, sr = librosa.load(audio_path, sr=None)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)
        
        return audio

    def _frame(self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
        signal_length = signal.shape[axis]
        if pad_end:
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = pad_axis = tuple([0 for i in range(2)])
                signal = torch.nn.functional.pad(torch.Tensor(signal), pad_axis, "constant", pad_value)
        frames=signal.unfold(axis, frame_length, frame_step)

        return frames

    def _audio_to_frames(self, samples):
        frame_size = DEFAULT_HOP_WIDTH
        samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode='constant')
        frames = self._frame(
            samples,
            frame_length=DEFAULT_HOP_WIDTH,
            frame_step=DEFAULT_HOP_WIDTH,
            pad_end=True
        )
        num_frames = len(samples) // frame_size
        times = np.arange(num_frames) / (DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH)

        return frames, times
    
    def _tokenize(self, midi):
        tokens = encode_midi(midi)

        return tokens
    
    def _get_random_length_segment(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        sample_length = random.randint(self.config.mel_length, input_length)
        start_length = random.randint(0, input_length - self.config.mel_length)

        for k in row.keys():
            if k in ['inputs', 'input_times']:
                new_row[k] = row[k][start_length:start_length+sample_length]
            else:
                new_row[k] = row[k]
        
        return new_row

    def _slice_segment(self, row):
        rows = []
        input_length = row['inputs'].shape[0]
        for split in range(0, input_length, self.config.mel_length):
            if split + self.config.mel_length >= input_length:
                continue
            new_row = {}
            for k in row.keys():
                if k in ['inputs', 'input_times']:
                    new_row[k] = row[k][split:split+self.config.mel_length]
                else:
                    new_row[k] = row[k]
            rows.append(new_row)
        
        if len(rows) == 0:
            return [row]
        return rows
    
    def _extract_target_sequence_with_indices(self, row):
        """Extract target sequence corresponding to audio token segment."""
        events = []
        target_start_time = row['input_times'][0]
        target_end_time = row['input_times'][-1]

        cur_time = 0
        cur_vel = 0

        for snote in row['targets']:
            if cur_time >= target_start_time and cur_time < target_end_time:
                events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
                events += _snote2events(snote=snote, prev_vel=cur_vel)
                events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)

            cur_time = snote.time
            cur_vel = snote.velocity
        row['targets'] = events

        return row
    
    def _target_to_int(self, row):
        row['targets'] = [e.to_int() for e in row['targets']]

        return row
    
    def _compute_spectrogram(self, ex):
        samples = torch.flatten(ex['inputs'])
        melspectrogram = MelSpectrogram(DEFAULT_NUM_MEL_BINS, DEFAULT_SAMPLE_RATE, FFT_SIZE, DEFAULT_HOP_WIDTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)

        ex['inputs'] = melspectrogram(samples.reshape(-1, samples.shape[-1])[:, :-1]).transpose(-1, -2).squeeze(0)
        
        return ex
    
    def _pad_length(self, row):
        inputs = row['inputs']
        end = row['end']
        targets = torch.from_numpy(np.array(row['targets'][:self.config.event_length])).to(torch.long).to('cpu')

        if inputs.shape[0] < self.config.mel_length:
            pad = torch.zeros(self.config.mel_length - inputs.shape[0], inputs.shape[1], dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat([inputs, pad], dim=0)

        if targets.shape[0] < self.config.event_length:
            eos = torch.ones(1, dtype=targets.dtype, device=targets.device) * TOKEN_END
            if self.config.event_length - targets.shape[0] - 1 > 0:
                pad = torch.ones(self.config.event_length - targets.shape[0] - 1, dtype=targets.dtype, device=targets.device) * TOKEN_PAD
                targets = torch.cat([targets, eos, pad], dim=0)
            else:
                targets = torch.cat([targets, eos], dim=0)

        if self.type == 'train':
            return {'inputs': inputs, 'targets': targets}
        else:
            return {'inputs': inputs, 'targets': targets, 'end': end}
    
    def _preprocess(self):
        for idx in range(len(self.midi_paths)):
            midi_path = str(self.midi_paths[idx])
            wav_path = str(self.wav_paths[idx])
            audio = self._load_audio(audio_path=wav_path, sample_rate=DEFAULT_SAMPLE_RATE)
            frames, frame_times = self._audio_to_frames(audio)
            encoded_midi = self._tokenize(str(midi_path))

            row = {'inputs': frames, 'input_times': frame_times, 'targets': encoded_midi, 'end': False}
            if self.type == 'train':
                row = self._get_random_length_segment(row)
            rows = self._slice_segment(row)

            for row in rows:
                if np.array_equal(row, rows[-1]):
                        row['end'] = True
                row = self._extract_target_sequence_with_indices(row)
                row = self._target_to_int(row)
                row = self._compute_spectrogram(row)
                row = self._pad_length(row)

                yield row

    def __len__(self): 
        return len(self.midi_paths)
    
    def __iter__(self):
        if self.type == 'train':
            return cycle(self._preprocess())
        else:
            return self._preprocess()