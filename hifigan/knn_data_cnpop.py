import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import librosa
import pyworld as pw

import pickle


def traverse_dir(
		root_dir,
		extension,
		amount=None,
		str_include=None,
		str_exclude=None,
		is_pure=False,
		is_sort=False,
		is_ext=True):

	file_list = []
	cnt = 0
	for root, _, files in os.walk(root_dir):
		for file in files:
			if file.endswith(extension):
				# path
				mix_path = os.path.join(root, file)
				# pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
				if root_dir.endswith("/"):
					pure_path = mix_path[len(root_dir):] if is_pure else mix_path
				else:
					pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

				# amount
				if (amount is not None) and (cnt == amount):
					if is_sort:
						file_list.sort()
					return file_list
				
				# check string
				if (str_include is not None) and (str_include not in pure_path):
					continue
				if (str_exclude is not None) and (str_exclude in pure_path):
					continue
				
				if not is_ext:
					ext = pure_path.split('.')[-1]
					pure_path = pure_path[:-(len(ext)+1)]
				file_list.append(pure_path)
				cnt += 1
	if is_sort:
		file_list.sort()
	return file_list


def get_data_loaders(args, whole_audio=False):
	# knn vectors, and f0 (from audio)
	
	assert args.data.block_size == args.data.hop_size
	data_train = AudioDataset(
		args.data.train_path,
		args.data.train_feat_path,
		waveform_sec=args.data.duration,
		hop_size=args.data.hop_size,
		sample_rate=args.data.sampling_rate,
		whole_audio=whole_audio)
	
	loader_train = torch.utils.data.DataLoader(
		data_train,
		batch_size=args.train.batch_size if not whole_audio else 1,
		shuffle=False,
		num_workers=8,
		pin_memory=True
	)
	data_valid = AudioDataset(
		args.data.valid_path,
		args.data.valid_feat_path,
		waveform_sec=args.data.duration,
		hop_size=args.data.hop_size,
		sample_rate=args.data.sampling_rate,
		whole_audio=True)
	loader_valid = torch.utils.data.DataLoader(
		data_valid,
		batch_size=1,
		shuffle=False,
		num_workers=1,
		pin_memory=True
	)
	return loader_train, loader_valid 






class AudioDataset(torch.utils.data.Dataset):
	# there is no need to take mel-spectrogram like in HiFi-GAN. We need 1. audio 2. corresponding wavLM knn-features. Now get f0 from audio.
	def __init__(
		self,
		audio_root_path,
		feat_root_path,
		waveform_sec,
		hop_size,
		sample_rate,
		whole_audio=False
	):

		super().__init__()
		
		self.waveform_sec = waveform_sec
		self.sample_rate = sample_rate
		self.hop_size = hop_size
		self.path_root = audio_root_path
		
		# deciding audio file ext automatically, but require that at most one of ".wav", ".mp3", ".flac" present
		audio_file_ext = None
		from pathlib import Path
		for temp_ext in [".wav", ".flac", ".mp3"]:
			if len(list(Path(audio_root_path).rglob("**/*" + temp_ext))) != 0:
				print("Set audio_file_ext to", temp_ext)
				audio_file_ext = temp_ext
				break
				
		if audio_file_ext is None:
			import sys
			sys.exit("No audio files with .wav, .flac, or .mp3 extension found")
		
		self.paths = traverse_dir(
			# os.path.join(path_root, 'audio'),
			audio_root_path,
			extension=audio_file_ext,
			is_pure=True,
			is_sort=True,
			# is_ext=False
			is_ext=True
		)
		
		self.feat_path_root = feat_root_path
		self.feat_paths = traverse_dir(
			# os.path.join(path_root, 'audio'),
			feat_root_path,
			extension=".pt",
			is_pure=True,
			is_sort=True,
			# is_ext=False
			is_ext=True
		)
		
		self.audio_file_ext = audio_file_ext
		
		# in case there are more features than audio (in the case of data subset)
		if len(self.paths) < len(self.feat_paths):
			# swap extension (to whatever audio ext), intersect, then swap back extension (.pt)
			# print(len(self.feat_paths))
			self.temp_feat_paths = set(item[:-3] + audio_file_ext for item in self.feat_paths).intersection(set(self.paths))
			
			# print(len(self.temp_feat_paths))
			self.feat_paths = sorted(item[:-len(audio_file_ext)] + ".pt" for item in list(self.temp_feat_paths))
			
		# if whole_audio:
			# print(self.feat_paths[:2], self.paths[:2], len(self.feat_paths), len(self.paths))
			# import sys
			# sys.exit()
		
		assert all(str(self.paths[i]).split("/")[-1][:-len(audio_file_ext)] + ".pt" == str(self.feat_paths[i]).split("/")[-1] for i in range(len(self.paths)))
		
		# print(self.paths, path_root)
		# import sys
		# sys.exit()
		self.whole_audio = whole_audio
		
	def __getitem__(self, file_idx):
		name = self.paths[file_idx]
		
		# check duration. if too short, then skip
		duration = librosa.get_duration(
			# filename=os.path.join(self.path_root, 'audio', name) + '.wav', 
			# filename=os.path.join(self.path_root, name) + '.wav', 
			filename=os.path.join(self.path_root, name), 
			sr=self.sample_rate)
			
		if duration < (self.waveform_sec + 0.1):
			return self.__getitem__(file_idx+1)
		
		# get item
		return self.get_data(name, duration)

	def get_data(self, name, duration):
		# path
		# path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
		# path_mel   = os.path.join(self.path_root, 'mel', name) + '.npy'
		path_audio = os.path.join(self.path_root, name)
		# path_mel   = path_audio.replace(".wav", ".npy")
		path_mel = os.path.join(self.feat_path_root, name[:-len(self.audio_file_ext)] + ".pt")

		# load nearest_nbr info
		with open(path_mel, 'rb') as handle:
			feat_dict = pickle.load(handle)
			# start_index, end_index = feat_dict["slice"]
			nearest_nbrs = feat_dict["nearest_nbrs"]
			
		# take the mean of first 4 nearest nbrs
		from pathlib import Path
		mel = torch.tensor(np.load(str(Path(path_mel).parent/"pool.npy"), mmap_mode = 'r')[nearest_nbrs[:, :4]]).mean(dim=1).half().float()


		# (seq_len, dim)
		# print(mel.shape)
		# import sys
		# sys.exit()
		
		# self.waveform_sec = waveform_sec
		# self.sample_rate = sample_rate
		# self.hop_size = hop_size
		waveform_sec = duration if self.whole_audio else self.waveform_sec
		
		# print(mel.size(0), duration*self.sample_rate/self.hop_size)
		if abs(mel.size(0) - duration*self.sample_rate/self.hop_size) >= 1:
			print(mel.shape, duration*self.sample_rate/self.hop_size)
			import sys
			sys.exit()
		
		
		import math
		frames_per_seg = math.ceil(waveform_sec * self.sample_rate / self.hop_size)
		# we already ensure that audio is longer than waveform_sec in __getitem__, hence directly select indices
		if self.whole_audio:
			mel_start = 0
		else:
			mel_start = random.randint(0, mel.size(0) - frames_per_seg - 1)
			
		# (T, mel_F)
		mel = mel[mel_start:mel_start + frames_per_seg, :]
		

		
		
		idx_from = mel_start * self.hop_size / self.sample_rate
		# renew waveform_sec so that it is a multiplier of hop_size
		waveform_sec = (frames_per_seg * self.hop_size) / self.sample_rate

		# load audio
		
		audio, sr = librosa.load(
				path_audio, 
				sr=None, 
				offset=idx_from,
				duration=waveform_sec)
		assert sr == self.sample_rate
		
		
		
		# print(audio.shape, path_audio, self.sample_rate, idx_from, waveform_sec)
		# import sys
		# sys.exit()
		frame_resolution = (self.hop_size / self.sample_rate)
		'''
		# clip audio into N seconds
		frame_rate_inv = 1/frame_resolution
		audio = audio[...,:audio.shape[-1]//self.hop_size*self.hop_size]

		mel_frame_len = int(waveform_sec*frame_rate_inv)

		# mel
		st = int(idx_from*frame_rate_inv)
		audio_mel_ = np.load(path_mel)
		audio_mel = audio_mel_[st:st+mel_frame_len]
		audio_mel = torch.from_numpy(audio_mel).float()
		'''
		
		# extract f0
		f0, _ = pw.dio(
			audio.astype('double'), 
			self.sample_rate, 
			f0_floor=65.0, 
			f0_ceil=1047.0, 
			channels_in_octave=2, 
			frame_period=(1000*frame_resolution))
		
		# f0 = f0.astype('float')[:audio_mel.size(0)]
		f0 = f0.astype('float')[:mel.size(0)]
		f0_hz = torch.from_numpy(f0).float().unsqueeze(-1)
		f0_hz[f0_hz<80]*= 0

		# import sys
		# sys.exit()
		# out 
		audio = torch.from_numpy(audio).float()
		# assert sr == self.sample_rate

		# print(mel.shape, audio.shape, f0_hz.shape)
		# import sys
		# sys.exit()

		# return dict(audio=audio, f0=f0_hz, mel=audio_mel, name=name)
		return dict(audio=audio, f0=f0_hz, mel=mel, name=name)

	def __len__(self):
		return len(self.paths)


