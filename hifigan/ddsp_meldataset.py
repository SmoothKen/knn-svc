import math
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read

import pickle
# def play_sequence(audio_chunk, f_s = 48000):
	# import sounddevice as sd
	# sd.play(audio_chunk, f_s, blocking = True)
	


def load_wav(full_path):
	#sampling_rate, data = read(full_path)
	#return data, sampling_rate
	data, sampling_rate = librosa.load(full_path, sr=None)
	return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
	return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
	return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
	return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
	return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
	output = dynamic_range_compression_torch(magnitudes)
	return output


def spectral_de_normalize_torch(magnitudes):
	output = dynamic_range_decompression_torch(magnitudes)
	return output


mel_basis = {}
hann_window = {}

class LogMelSpectrogram(torch.nn.Module):
	def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
		super().__init__()
		self.melspctrogram = torchaudio.transforms.MelSpectrogram(
			sample_rate=sampling_rate,
			n_fft=n_fft,
			win_length=win_size,
			hop_length=hop_size,
			center=center,
			power=1.0,
			norm="slaney",
			onesided=True,
			n_mels=num_mels,
			mel_scale="slaney",
			f_min=fmin,
			f_max=fmax
		)
		self.n_fft = n_fft
		self.hop_size = hop_size

	def forward(self, wav):
		wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
		mel = self.melspctrogram(wav)
		logmel = torch.log(torch.clamp(mel, min=1e-5))
		return logmel



class LinearSpectrogram(torch.nn.Module):
	def __init__(self, n_fft, hop_size, win_size, center=False):
		super().__init__()
		self.linearspctrogram = torchaudio.transforms.Spectrogram(
			n_fft=n_fft,
			win_length=win_size,
			hop_length=hop_size,
			center=center,
			power=None
		)
		self.n_fft = n_fft
		self.hop_size = hop_size

	def forward(self, wav):
		wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
		spec = self.linearspctrogram(wav)
		spec = spec[:, :-1, :]
		# spec = torch.cat([spec.real, spec.imag], dim = 1)
		
		return torch.abs(spec)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, stop_at_spectrogram = False):
	if torch.min(y) < -1.:
		print('min value is ', torch.min(y))
	if torch.max(y) > 1.:
		print('max value is ', torch.max(y))

	global mel_basis, hann_window
	if fmax not in mel_basis:
		mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
		mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
		hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

	# print("Padding by", int((n_fft - hop_size)/2), y.shape)
	# pre-padding
	n_pad = hop_size - ( y.shape[1] % hop_size )
	y = F.pad(y.unsqueeze(1), (0, n_pad), mode='reflect').squeeze(1)
	# print("intermediate:", y.shape)

	y = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
	y = y.squeeze(1)
	
	spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
					  center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)


	print("!!!", spec.shape)
	if stop_at_spectrogram:
		spec = spec[:, :-1, :]
		spec = torch.cat([spec.real, spec.imag], dim = 1)
		# print(torch.cat([spec.real, spec.imag], dim = 1).shape)
		# import sys
		# sys.exit()
		print("---", spec.shape)
		return spec	


	spec = spec.abs().clamp_(3e-5)
	# print("Post: ", y.shape, spec.shape)

	spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
	spec = spectral_normalize_torch(spec)

	print("???", spec.shape)
	return spec


def get_dataset_filelist(a):
	train_df = pd.read_csv(a.input_training_file)
	valid_df = pd.read_csv(a.input_validation_file)
	return train_df, valid_df


def interpolate_f0(data):
	
	data = np.reshape(data, (data.size, 1))

	vuv_vector = np.zeros((data.size, 1),dtype=np.float32)
	vuv_vector[data > 0.0] = 1.0
	vuv_vector[data <= 0.0] = 0.0

	ip_data = data

	frame_number = data.size
	last_value = 0.0
	for i in range(frame_number):
		if data[i] <= 0.0:
			j = i + 1
			for j in range(i + 1, frame_number):
				if data[j] > 0.0:
					break
			if j < frame_number - 1:
				if last_value > 0.0:
					step = (data[j] - data[i - 1]) / float(j - i)
					for k in range(i, j):
						ip_data[k] = data[i - 1] + step * (k - i + 1)
				else:
					for k in range(i, j):
						ip_data[k] = data[j]
			else:
				for k in range(i, frame_number):
					ip_data[k] = last_value
		else:
			ip_data[i] = data[i]
			last_value = data[i]

	return ip_data, vuv_vector

'''

def upsample(signal, factor, mode = "nearest"):
	import torch.nn as nn
	return nn.functional.interpolate(signal, size=signal.shape[-1] * factor, mode = mode)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
	
	
	import torch
	n_harm = amplitudes.shape[-1]
	pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)[None, None, :]
	aa = (pitches < sampling_rate / 2).float() + 1e-7
	
	
	assert amplitudes.shape == aa.shape, [amplitudes.shape, aa.shape]
	return amplitudes * aa






def get_bulk_dsp_choral(f0, amp, sample_rate = 16000, hop_size = 320, dsp_type = "sin"):

	f0 = upsample(f0.transpose(1, 2), hop_size).transpose(1, 2)
	# n_harmonic = 256
	
	# n_harmonic = 1
	# n_harmonic_amp_multipliers = 1/(torch.arange(1, n_harmonic/10+1, 0.1, device = self.device)**2)[None, None, :]
	# n_harmonic_amp_multipliers = 1/(torch.arange(1, n_harmonic/10+1, 0.1, device = self.device))[None, None, :]
	
	# n_harmonic_amp_multipliers = torch.ones((n_harmonic,), device = f0.device)[None, None, :]
	amp = upsample(amp.transpose(1, 2), hop_size, mode = "linear").transpose(1, 2)
	# amp = (f0 != 0)*amp
	# print(f0.shape, amp.shape)
	
	
	n_harmonic_amp_multipliers = amp
	n_harmonic = amp.shape[-1]
	
	
	# import sys
	# sys.exit()

	
	
	phase = torch.cumsum(f0.double() / sample_rate, dim = 1)
	import math
	phase = (2 * math.pi * (phase - torch.round(phase))).float()
	phases = phase * torch.arange(1, n_harmonic + 1, device = phase.device)
	
	# print(phases.shape, n_harmonic_amp_multipliers.shape)
	# import sys
	# sys.exit()
	# assert phases[-1].shape == n_harmonic_amp_multipliers[-1].shape
	amp = remove_above_nyquist(n_harmonic_amp_multipliers*torch.ones(phases.shape, device = phases.device), f0, sample_rate)
	
	# -> (B, T_upsampled, 1)
	ddsp_signal = (torch.sin(phases) * amp).sum(-1, keepdim = True)

	return ddsp_signal

'''
import sys, os
from pathlib import Path
sys.path.append(Path(os.path.abspath(__file__)))
from ddsp_prematch_dataset import get_bulk_dsp_choral


def play_sequence(audio_chunk, f_s = 16000):
	import sounddevice as sd
	sd.play(audio_chunk, f_s, blocking = True)


def write_audio(filename, waveform, sample_rate):
	
	# first convert as we may need it to become bytes later
	import torch
	if isinstance(waveform, torch.Tensor):
		waveform = waveform.detach().cpu().numpy()

	
	import numpy as np
	# print(waveform.shape, np.max(waveform), np.min(waveform))

	
	# convert to int32 if it is [-1, 1] float
	if waveform.dtype == np.float32 or waveform.dtype == np.float64:
		
		# ensure it is in [-1, 1]
		waveform_abs_max = np.max(np.abs(waveform))
		if waveform_abs_max > 1:
			waveform = waveform/waveform_abs_max
	
		
		
		waveform = waveform * (2 ** 31 - 1)   
		waveform = waveform.astype(np.int32)
	else:
		assert waveform.dtype == np.int32
		
		

	if filename.endswith(".wav"):
		import soundfile as sf
		sf.write(filename, waveform.T, samplerate = sample_rate, subtype = 'PCM_32')
		
	else:
		
		
		if waveform.ndim == 2:
			if waveform.shape[0] in {1, 2}:
				waveform = waveform.T
				
			channels = waveform.shape[1]
		elif waveform.ndim == 1:
			channels = 1
		else:
			import sys
			sys.exit("Bad audio array shape")
		
			
		
		from pydub import AudioSegment
		song = AudioSegment(waveform.tobytes(), frame_rate=sample_rate, sample_width=4, channels=channels)
		
		assert filename.split(".")[-1] in {"mp3", "flac"}
		
		song.export(filename, format=filename.split(".")[-1], bitrate="320k")




class MelDataset(torch.utils.data.Dataset):
	def __init__(self, hps, segment_size, n_fft, num_mels,
				 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
				 device=None, fmax_loss=None, fine_tuning=False, audio_root_path=None, feat_root_path=None, use_alt_melcalc=False):
		
		self.audio_root_path = Path(audio_root_path)
		self.feat_root_path = Path(feat_root_path)
		
		audio_paths = []
		# , ".mp3"
		for temp_ext in [".flac", ".wav"]:
			audio_paths += list(os.path.relpath(item, self.audio_root_path) for item in sorted(self.audio_root_path.rglob('*' + temp_ext)))
			print(self.audio_root_path, temp_ext, len(list(sorted(self.audio_root_path.rglob('**/*' + temp_ext)))))
			'''
			if len(audio_paths) != 0:
				audio_file_ext = temp_ext
				break
			'''
				
		if len(audio_paths) == 0:
			import sys
			sys.exit("No audio files found")
		
		
		feat_paths = list(os.path.relpath(item, self.feat_root_path) for item in sorted(self.feat_root_path.rglob('*.pt')))
		
		# ensure Cantoria_EJB2_S_resampled_16000.wav and Cantoria_EJB2_B_resampled_16000.wav appears
		if not split:
			temp_feat_paths = []
			temp_audio_paths = []
			
			special_required_feat_paths = []
			special_required_audio_paths = []
			for item_idx, item in enumerate(feat_paths):
				#  or "Cantoria_EJB2_B_resampled_16000.pt" in item
				if "Cantoria_EJB2_S_resampled_16000.pt" in item:
					special_required_feat_paths.append(item)
					special_required_audio_paths.append(audio_paths[item_idx])
				else:
					temp_feat_paths.append(item)
					temp_audio_paths.append(audio_paths[item_idx])
			
		
			feat_paths = special_required_feat_paths + temp_feat_paths
			audio_paths = special_required_audio_paths + temp_audio_paths
			# print(feat_paths)
			# print(audio_paths)
			# import sys
			# sys.exit()
		
		
		
		if len(audio_paths) != len(feat_paths) or not all(str(audio_paths[i]).split("/")[-1].replace(".wav", ".pt").replace(".mp3", ".pt").replace(".flac", ".pt") == str(feat_paths[i]).split("/")[-1] for i in range(len(audio_paths))):
			print(len(audio_paths), len(feat_paths))
			print(audio_paths[0], feat_paths[0])
			import sys
			sys.exit()
		self.audio_files = pd.DataFrame({'audio_path': audio_paths, "feat_path": feat_paths})
		
		# self.audio_files = training_files
		if shuffle:
			self.audio_files = self.audio_files.sample(frac=1, random_state=1234)
		self.segment_size = segment_size
		self.sampling_rate = sampling_rate
		self.split = split
		self.n_fft = n_fft
		self.num_mels = num_mels
		self.hop_size = hop_size
		self.win_size = win_size
		self.fmin = fmin
		self.fmax = fmax
		self.fmax_loss = fmax_loss
		self.cached_wav = None
		self.n_cache_reuse = n_cache_reuse
		self._cache_ref_count = 0
		self.device = device
		self.fine_tuning = fine_tuning
		self.audio_root_path = Path(audio_root_path)
		self.feat_root_path = Path(feat_root_path)
		self.alt_melspec = LogMelSpectrogram(n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
		
		self.alt_linearspec = LinearSpectrogram(n_fft, hop_size, win_size)
		self.use_alt_melcalc = use_alt_melcalc

		self.hps = hps
		
		# test
		# not 
		'''
		if not self.split:
			import numpy as np
			choice = np.random.choice(np.arange(self.__len__()))
			print(choice)
			self.__getitem__(choice)
			import sys
			sys.exit()
		'''

		
	def __getitem__(self, index):
		row = self.audio_files.iloc[index]
		assert self._cache_ref_count == 0
		assert self.fine_tuning
		
		if self._cache_ref_count == 0:
			audio, sampling_rate = load_wav(self.audio_root_path/row.audio_path)
			if not self.fine_tuning:
				audio = normalize(audio) * 0.95
			self.cached_wav = audio
			if sampling_rate != self.sampling_rate:
				raise ValueError("{} SR doesn't match target {} SR".format(
					sampling_rate, self.sampling_rate))
			self._cache_ref_count = self.n_cache_reuse
		else:
			audio = self.cached_wav
			self._cache_ref_count -= 1

		audio = torch.tensor(audio, dtype=torch.float32)
		audio = audio.unsqueeze(0)
		assert self.fine_tuning

		if not self.fine_tuning:
			if self.split:
				if audio.size(1) >= self.segment_size:
					max_audio_start = audio.size(1) - self.segment_size
					audio_start = random.randint(0, max_audio_start)
					audio = audio[:, audio_start:audio_start+self.segment_size]
				else:
					audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

			if self.use_alt_melcalc:
				mel = self.alt_melspec(audio)
			else:
				mel1 = mel_spectrogram(audio, self.n_fft, self.num_mels,
								 self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
								 center=False)
			
			mel = mel.permute(0, 2, 1) # (1, dim, seq_len) --> (1, seq_len, dim)
		else:
			# mel = torch.load(self.feat_root_path/row.feat_path, map_location='cpu').float() 
			
			with open(str(self.feat_root_path/row.feat_path), 'rb') as handle:
				feat_dict = pickle.load(handle)
				# start_index, end_index = feat_dict["slice"]
				nearest_nbrs = feat_dict["nearest_nbrs"]
				nearest_nbrs_f0_priority = feat_dict["nearest_nbrs_f0_priority"]
				amp_ratio = torch.tensor(feat_dict["amp_ratio"])
				
				
			# .mean(dim=1).half().float()
			mel = torch.tensor(np.load(str((self.feat_root_path/row.feat_path).parent/"pool.npy"), mmap_mode = 'r')[nearest_nbrs[:, :4]]).mean(dim=1)
			
			
			# extra_mel = torch.tensor(np.load(str((self.feat_root_path/row.feat_path).parent/"pool_spec.npy"), mmap_mode = 'r')[nearest_nbrs_f0_priority[:, :4]])
			harmonics_out_feats = torch.tensor(np.load(str((self.feat_root_path/row.feat_path).parent/"pool_harmonics.npy"), mmap_mode = 'r')[nearest_nbrs_f0_priority[:, :4]])
			
				
			# harmonics_out_feats_weighted = torch.mean(harmonics_out_feats*amp_ratio[..., None], dim = 1)
			
			# harmonics_out_feats_weighted = harmonics_out_feats[:, 0]*amp_ratio[:, 0, None]
			
			# from random import randrange
			# print(randrange(10))

			# random_indices = np.random.randint(4, size=(len(harmonics_out_feats),))

			random_indices = torch.randint(low=0, high=harmonics_out_feats.shape[1], size=(len(harmonics_out_feats),))[:, None].to(harmonics_out_feats.device)
			harmonics_out_feats_weighted = (torch.gather(harmonics_out_feats, index = random_indices[..., None].repeat(1, 1, harmonics_out_feats.shape[-1]), dim = 1)*torch.gather(amp_ratio, index = random_indices, dim = 1)[..., None]).squeeze(1)
			
			
			
			
			# merge to cut together, then split at the end
			num_of_mel_bins = mel.shape[-1]
			mel = torch.cat([mel, harmonics_out_feats_weighted], dim = -1)
			
			
			if len(mel.shape) < 3:
				mel = mel.unsqueeze(0) # (1, seq_len, dim)

			if self.split:
				frames_per_seg = math.ceil(self.segment_size / self.hop_size)

				if audio.size(1) >= self.segment_size:
					mel_start = random.randint(0, mel.size(1) - frames_per_seg - 1)
					mel = mel[:, mel_start:mel_start + frames_per_seg, :]
					audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
				else:
					mel = torch.nn.functional.pad(mel, (0, 0, 0, frames_per_seg - mel.size(2)), 'constant')
					audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')


			if self.split:
				import pyworld as pw
				# print(mel.shape, audio.shape)
				# import sys
				# sys.exit()
				f0, _ = pw.harvest(audio.squeeze().numpy().astype(np.float64), self.hps.sampling_rate, f0_floor=65.0, f0_ceil=1047.0, frame_period=self.hps.hop_size / self.hps.sampling_rate * 1000)

				f0 = torch.from_numpy(f0).float()
				f0[f0<80] *= 0	
			else:
				# load from pt
				with open(str(self.feat_root_path/row.feat_path), 'rb') as handle:
					feat_dict = pickle.load(handle)
					# start_index, end_index = feat_dict["slice"]
					f0 = feat_dict["f0"]
					
			# (23,)
			# print(f0.shape, row.audio_path)
			# print(f0)
			# f0, _ = interpolate_f0(f0)
			# print(f0.squeeze())
			# print(f0.shape, mel.shape)
			
			# f0 = f0.astype('float')[:audio_mel.size(0)]
			f0 = f0[:mel.size(1)]
			
			
			
			# import sys
			# sys.exit()
			

		if self.use_alt_melcalc:
			mel_loss = self.alt_melspec(audio)
		else:
			mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
								   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
								   center=False)
		
		# mel -> [frames, num_mels], f0 -> [1, frames]
		# print(mel.shape, fake_audio.shape, audio.shape, f0.shape, mel_loss.shape)
		
		# split the mel and harmonics_amp
		harmonics_out_feats_weighted = mel[:, :, num_of_mel_bins:]
		mel = mel[:, :, :num_of_mel_bins]
		assert harmonics_out_feats_weighted.shape[-1] == 49, harmonics_out_feats_weighted.shape[-1]
		
		# import sys, os
		# sys.path.insert(1, os.path.realpath(os.path.pardir))
		# import sys
		# sys.exit()
		
		# assert self.hop_size == 320
		# dsp_signal = get_bulk_dsp_choral(f0[None, :, None], harmonics_out_feats_weighted, sample_rate = self.sampling_rate, hop_size = self.hop_size, dsp_type = "sin").reshape(-1)
		# play_sequence(dsp_signal, f_s = self.sampling_rate)
		# write_audio("/tmp/temp_4.mp3", dsp_signal, 16000)
		
		# mel = torch.log(torch.clamp(mel*torch.exp(torch.max(mel_loss))/torch.max(mel), min=1e-5))
		
		# print(torch.max(mel), torch.min(mel), torch.max(audio), torch.min(audio), torch.max(mel_loss), torch.min(mel_loss))
		# import sys
		# sys.exit()
		
		# return mel.squeeze(), audio.squeeze(0), str(row.audio_path), mel_loss.squeeze(), f0.unsqueeze(0)
		
		# return mel.squeeze(), audio.squeeze(0), str(row.audio_path), mel_loss.squeeze(), dsp_signal.unsqueeze(0)
		return mel.squeeze(), audio.squeeze(0), str(row.audio_path), mel_loss.squeeze(), f0.unsqueeze(0), harmonics_out_feats_weighted.squeeze()

	def __len__(self):
		return len(self.audio_files)
