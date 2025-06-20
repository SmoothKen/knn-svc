import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastprogress.fastprogress import master_bar, progress_bar
from torch import Tensor

from hubconf import wavlm_large

DOWNSAMPLE_FACTOR = 320

global feature_cache
feature_cache = {}
global synthesis_cache
synthesis_cache = {}


# assume col_names is the same as row_names 
def plot_matrix(mat, row_names = None, col_names = None, title = "", x_axis = "", y_axis = "", fig = None, fig_row = 2, fig_col = 1, reverse = False):

	import plotly.express as px
	show_on_screen = (fig is None)
	if show_on_screen:
		fig = px.imshow(mat, text_auto=True, x=col_names, y=row_names, aspect='auto', color_continuous_scale='Bluered_r')
		
		
	else:
		import plotly.graph_objects as go
		fig_imshow = px.imshow(mat, text_auto=True, x=col_names, y=row_names, aspect='auto', color_continuous_scale='Bluered_r')
		fig.add_trace(go.Heatmap(fig_imshow.data[0]),row=fig_row, col=fig_col)
		
		# fig.update_layout(coloraxis_showscale=True, coloraxis=dict(colorbar_len=0.5, colorbar_y=0.80))
		fig.update_layout(coloraxis_showscale=False)
	
	fig.update_layout(
		title=title,
		xaxis_title=x_axis,
		yaxis_title=y_axis,
		margin={"l":40, "r":40, "t":40, "b":40},
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		autosize=True,
		template="simple_white"
	)

	if show_on_screen:
		fig.show()	
	else:
		return fig


def make_librispeech_df(root_path: Path) -> pd.DataFrame:
	all_files = []
	folders = ['train-clean-100', 'dev-clean']
	print(f"[LIBRISPEECH] Computing folders {folders}")
	for f in folders:
		all_files.extend(list((root_path/f).rglob('**/*.flac')))
	speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
	df = pd.DataFrame({'path': all_files, 'speaker': speakers})
	return df


# path -> spk_folder, dataset_root_dir -> folder that contains all spk_folders
# duration_limit in second

# f0_dir_parent -> the parent dir of f0_dir
def get_complete_spk_pool(path: Path, wavlm: nn.Module(), match_weights: Tensor, synth_weights: Tensor, f0_dir_parent, device = "cuda", duration_limit = None, vad_trigger_level = 0):
	"""load and wavlm process all flac for any given spk"""

	matching_pool = dict()
	synth_pool = dict()
	spec_synth_pool = dict()
	audio_synth_pool = dict()
	f0_pool = dict()
	harmonics_amp_synth_pool = dict()
	if os.path.isfile(path) and os.path.splitext(path)[-1] in {".flac", ".wav", ".mp3"}:
		print("Info: Processing a Single File")
		uttrs_from_same_spk = [path]
	else:
		uttrs_from_same_spk = sorted(list(path.rglob('**/*.flac')) + list(path.rglob('**/*.wav')))

	
	if len(uttrs_from_same_spk) == 0:
		print("Bad directory", path)
		import sys
		sys.exit()
	

	STFT_OP = torchaudio.transforms.Spectrogram(n_fft = 400, hop_length = DOWNSAMPLE_FACTOR, center = True, power = 1)
	
	
	# ensure rglob above sorted to avoid result variance
	accumulated_duration_so_far = 0


	for pth in uttrs_from_same_spk:
		# print(pth)
		
		x, sr = torchaudio.load(pth)
		# print(torch.max(torch.abs(x)), torch.mean(torch.abs(x)))
		if x.shape[0] > 1:
			print("WARNING, converting to mono")
			x = torch.mean(x, dim = 0, keepdim = True)
		# print(x.shape)
		# import sys
		# sys.exit()
		assert sr == 16000
		
		feats = get_full_features(x, sr, wavlm, device)
		
		# weights here decide weight apply to each wavlm layer. (e.g. if [0, 1, 0....] means we only take the second layer
		
		# print(feats.shape, match_weights.shape)
		# import sys
		# sys.exit()
		
		matching_pool[str(pth)] = ( feats*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)
		synth_pool[str(pth)] = ( feats*synth_weights[:, None] ).sum(dim=0) # (seq_len, dim)
		
		assert len(matching_pool[str(pth)]) == len(synth_pool[str(pth)])
	
		assert len(x.squeeze()) >= DOWNSAMPLE_FACTOR*len(matching_pool[str(pth)])
		audio_synth_pool[str(pth)] = x.squeeze()[:DOWNSAMPLE_FACTOR*len(matching_pool[str(pth)])].reshape(len(matching_pool[str(pth)]), DOWNSAMPLE_FACTOR)
		
		
		# [audio_len] -> [seq_len, dim]
		
		# audio_synth_pool[str(pth)] =
		spec = STFT_OP(x.squeeze()).T[:, :-1]
		assert spec.shape[0] >= synth_pool[str(pth)].shape[0]
		spec = spec[:synth_pool[str(pth)].shape[0]]
		
		print(spec.shape, synth_pool[str(pth)].shape)
		spec_synth_pool[str(pth)] = spec.to(synth_pool[str(pth)].device)
		# import sys
		# sys.exit()
		
		f0_path = os.path.join(f0_dir_parent, f"f0_cache_{str(sr)}_{DOWNSAMPLE_FACTOR}", os.path.relpath(pth, f0_dir_parent).replace(os.path.splitext(pth)[-1], ".npy"))
	
		if not os.path.isfile(f0_path):
			print(f"WARNING: {f0_path} not exists, generating...")
			f0_pool[str(pth)] = get_f0(x, sr)
			
			Path(f0_path).parent.mkdir(parents=True, exist_ok=True)
			np.save(f0_path, f0_pool[str(pth)])
		else:
			f0_pool[str(pth)] = torch.tensor(np.load(f0_path, allow_pickle = True))
		
		
		assert abs(len(f0_pool[str(pth)]) - len(synth_pool[str(pth)])) <= 1 and len(f0_pool[str(pth)]) >= len(synth_pool[str(pth)])
		f0_pool[str(pth)] = f0_pool[str(pth)][:len(synth_pool[str(pth)])]
		
		
		
		
		matching_harmonics = (f0_pool[str(pth)][:, None].to(spec_synth_pool[str(pth)].device))*(torch.arange(1, 50, device = spec_synth_pool[str(pth)].device)[None, :])
		assert len(matching_harmonics.shape) == 2 and 16000/(2*spec_synth_pool[str(pth)].shape[-1]) == 40, [len(matching_harmonics.shape), len(spec_synth_pool[str(pth)].shape), 16000/(2*spec_synth_pool[str(pth)].shape[-1])]
		
		
		interpolated_spec = F.interpolate(spec_synth_pool[str(pth)][None, :], scale_factor = 8, mode='linear').squeeze(0)
		# interpolated_spec = spec_synth_pool[str(pth)]
		matching_harmonics_indices = torch.round(torch.clamp((matching_harmonics*2*interpolated_spec.shape[-1]/16000), max = interpolated_spec.shape[-1])).to(int)

		harmonics_synth_temp = torch.gather(F.pad(interpolated_spec, (0, 1)), dim = -1, index = matching_harmonics_indices)
		
		harmonics_synth_temp[:, 1:][f0_pool[str(pth)] == 0] = 0
		harmonics_synth_temp[:, 0][f0_pool[str(pth)] == 0] = torch.max(spec_synth_pool[str(pth)], dim = 1)[0][f0_pool[str(pth)] == 0]
		
		harmonics_amp_synth_pool[str(pth)] = 0.0108*harmonics_synth_temp
	

		
		accumulated_duration_so_far += len(spec_synth_pool[str(pth)])*DOWNSAMPLE_FACTOR/sr
		if duration_limit is not None and accumulated_duration_so_far >= duration_limit:
			print(f"Duration Limit, cut at {accumulated_duration_so_far}/{duration_limit}")
			
			# import sys
			# sys.exit()
			
			break

			
	return matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_amp_synth_pool


def path2pools(path: Path, wavlm: nn.Module(), match_weights: Tensor, synth_weights: Tensor, device):
	"""Given a waveform `path`, compute the matching pool"""

	uttrs_from_same_spk = sorted(list(path.parent.rglob('**/*.flac')))
	uttrs_from_same_spk.remove(path)
	matching_pool = []
	synth_pool = []
	for pth in uttrs_from_same_spk:
		if pth in feature_cache and pth in synthesis_cache:
			matching_feats = feature_cache[pth].float() # (seq_len, dim)
			synth_feats = synthesis_cache[pth].float() # (seq_len, dim)
		else:
			feats = get_full_features(pth, wavlm, device)
			matching_feats = ( feats*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)
			synth_feats = ( feats*synth_weights[:, None] ).sum(dim=0) # (seq_len, dim)
			feature_cache[pth] = matching_feats.half().cpu()
			synthesis_cache[pth] = synth_feats.half().cpu()

		matching_pool.append(matching_feats.cpu())
		synth_pool.append(synth_feats.cpu())
	matching_pool = torch.concat(matching_pool, dim=0)
	synth_pool = torch.concat(synth_pool, dim=0)
	return matching_pool, synth_pool # (N, dim)


# , pth = None
def get_f0(x, sr):
	# print("Started", pth)
	import pyworld as pw
	f0, _ = pw.harvest(x.squeeze().numpy().astype(np.float64), sr, f0_floor=65.0, f0_ceil=1047.0, frame_period=DOWNSAMPLE_FACTOR / sr * 1000)
	f0 = torch.from_numpy(f0).float()
	f0[f0<80] *= 0	

	# print("Ended", pth)
	return f0
	# , f0[:features.shape[1]]


def upsample(signal, factor, mode = "nearest"):
	import torch.nn as nn
	# print(f"Using {mode} interpolation")
	# return nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
	if mode == "bicubic":
		# print(signal[:, :, None].shape)
		# import sys
		# sys.exit()
		
		
		return nn.functional.interpolate(signal[:, :, None], size=(1, signal.shape[-1] * factor), mode = mode).squeeze(2)
	else:
		return nn.functional.interpolate(signal, size=signal.shape[-1] * factor, mode = mode)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
	
	
	import torch
	n_harm = amplitudes.shape[-1]
	pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)[None, None, :]
	aa = (pitches < sampling_rate / 2).float() + 1e-7
	
	
	assert amplitudes.shape == aa.shape, [amplitudes.shape, aa.shape]
	return amplitudes * aa




# expect
# f0 (batch, seq_len ,1)
# amp (batch, seq_len, harmonics)
# output (batch, wav_len, 1)
# , device = "cpu"
def get_bulk_dsp_choral(f0, amp, sample_rate = 16000, hop_size = 320, dsp_type = "sin"):

	assert f0.device == amp.device, [f0.device, amp.device]
	f0 = upsample(f0.transpose(1, 2), hop_size).transpose(1, 2)
	# n_harmonic = 256
	
	# n_harmonic = 1
	# n_harmonic_amp_multipliers = 1/(torch.arange(1, n_harmonic/10+1, 0.1, device = self.device)**2)[None, None, :]
	# n_harmonic_amp_multipliers = 1/(torch.arange(1, n_harmonic/10+1, 0.1, device = self.device))[None, None, :]
	
	# n_harmonic_amp_multipliers = torch.ones((n_harmonic,), device = f0.device)[None, None, :]
	
	# amp = upsample(amp.transpose(1, 2), hop_size).transpose(1, 2)
	# amp = upsample(amp.transpose(1, 2), hop_size, mode = 'linear').transpose(1, 2)
	amp = upsample(amp.transpose(1, 2), hop_size, mode = 'bicubic').transpose(1, 2)

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

# f0 [B, T], the bth f0 candidate at time t
# amps [B, T, N]
def get_bulk_dsp(f0, amp, sample_rate = 16000, hop_size = 320, dsp_type = "sin", device = "cpu"):
	
	assert len(f0.shape) == 2 and len(amp.shape) == 3 and f0.shape[:2] == amp.shape[:2], [len(f0.shape), len(amp.shape), f0.shape[:2], amp.shape[:2]]
	
	import torch
	f0 = f0.to(device)
	amp = amp.to(device)
	
	amp[f0[..., None] == 0] = 0

	f0 = upsample(f0[..., None].transpose(1, 2), hop_size).transpose(1, 2)[..., 0]
	amp = upsample(amp.transpose(1, 2), hop_size).transpose(1, 2)


	'''
	n_harmonic = 256
	n_harmonic_amp_multipliers = 1/(torch.arange(1, n_harmonic/10+1, 0.1, device = f0.device)**2)[None, None, :]
	'''
	
	# import sys
	# sys.exit()
	
	# ensure starting from 0 phase
	if dsp_type == "sin":
		initial_phase = torch.zeros_like(f0[:, :1])
	elif dsp_type == "cos":
		import math
		# not multiplying 2*pi yet
		initial_phase = torch.ones_like(f0[:, :1])/4
	else:
		raise NotImplementedError
		
	phase = torch.cumsum(torch.cat([initial_phase.double(), f0.double() / sample_rate], dim = 1)[:, :-1], dim = 1)
	import math
	phase = (2 * math.pi * (phase - torch.round(phase))).float()
	n_harmonic = amp.shape[-1]
	# print(phase.shape, torch.arange(1, n_harmonic + 1, device = f0.device).shape)
	
	# print(phase.shape)
	# [B, T] -> [B, T, 1] -> [B, T, N]
	phases = phase[..., None] * torch.arange(1, n_harmonic + 1, device = f0.device)[None, None, :]


	# assert phases[-1].shape == n_harmonic_amp_multipliers[-1].shape
	amp = remove_above_nyquist(amp, f0[..., None], sample_rate)
	
	# -> (B, T_upsampled, 1)
	# print(phases[:, :1])
	ddsp_signal = (torch.sin(phases) * amp).sum(-1, keepdim = False)
	
	# print(phases.shape, ddsp_signal.shape)
	# import sys
	# sys.exit()
	
	# return shape [B, T]
	return ddsp_signal




@torch.inference_mode()
def get_full_features(x, sr, wavlm, device):

	
	# print(x.shape)
	# import sys
	# sys.exit()

	
	# print(x.shape)
	# import sys
	# sys.exit()
	features_list = []
	
	start = 0
	while start < x.shape[-1]:
		
		x_chunk = x[..., start:start+30*sr]
		# if too short, discard
		if x_chunk.shape[-1] <= 0.02*sr:
			break	
		# This does not work i.t.o the hifigan training.
		# x = F.pad(x, (DOWNSAMPLE_FACTOR//2, DOWNSAMPLE_FACTOR - DOWNSAMPLE_FACTOR//2), value=0)
		# This does.
		n_pad = DOWNSAMPLE_FACTOR - (x_chunk.shape[-1] % DOWNSAMPLE_FACTOR)
		x_chunk = F.pad(x_chunk, (0, n_pad), value=0)

		# extract the representation of each layer
		wav_input_16khz = x_chunk.to(device)
		rep, layer_results = wavlm.extract_features(wav_input_16khz, output_layer=wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
		features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
		features_list.append(features)
		
		start = start+30*sr

	features = torch.cat(features_list, dim=1)
	# torch.Size([25, 1818, 1024]) -> (wavlm_layer, time_steps, feature_dim)
	# print(features.shape, f0.shape)
	# import sys
	# sys.exit()
	return features

'''
def fast_cosine_dist(source_feats, matching_pool):
	source_norms = torch.norm(source_feats, p=2, dim=-1)
	matching_norms = torch.norm(matching_pool, p=2, dim=-1)
	dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
	dotprod /= 2

	
	dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
	if torch.sum(torch.isnan(dists)) > 0:
		print("containing nan")
		import sys
		sys.exit()
	return dists
'''

@torch.inference_mode()
def extract(df: pd.DataFrame, wavlm: nn.Module, device, ls_path: Path, out_path: Path, synth_weights: Tensor, match_weights: Tensor):
	
	pb = progress_bar(df.iterrows(), total=len(df))

	for i, row in pb:
		rel_path = Path(row.path).relative_to(ls_path)
		targ_path = (out_path/rel_path).with_suffix('.pt')
		if args.resume:
			if targ_path.is_file(): continue
		# if targ_path.is_file(): continue
		os.makedirs(targ_path.parent, exist_ok=True)

		if Path(row.path) in feature_cache:
			source_feats = feature_cache[Path(row.path)].float()
		else:
			source_feats = get_full_features(row.path, wavlm, device)
			source_feats = ( source_feats*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)

		# get the pool of utterance (features) from all flac of the same speaker
		matching_pool, synth_pool = path2pools(row.path, wavlm, match_weights, synth_weights, device)

		if not args.prematch:
			out_feats = source_feats.cpu()
		else:
			
			dists = fast_cosine_dist(source_feats.cpu(), matching_pool.cpu()).cpu()
			best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
			
			
			# test
			best = dists.topk(k=args.topk, dim=-1, largest=False)
			out_feats = synth_pool[best.indices].mean(dim=1)
			old_target_feature_path = str(targ_path).replace("cached", "cached_old")
			old_out_feats = torch.load(old_target_feature_path).to(out_feats)
			print(old_target_feature_path, source_feats.shape, matching_pool.shape, out_feats.shape)
			
			print(torch.max(torch.abs(old_out_feats - out_feats.half())))
			
			if "5652-19215-0000.pt" in str(targ_path):
				import sys
				sys.exit()
			else:
				continue
				
			
			

			# save synth_pool and best indices so that k becomes flexible during experiment time.
			out_feats = source_feats.cpu()
			# out_feats = synth_pool[best.indices].mean(dim=1) # (N, dim)

		# save matched sequence
		if i < 3: print("Feature has shape: ", out_feats.shape, flush=True)
		# 3. save
		torch.save(out_feats.cpu().half(), str(targ_path))

		
		if hasattr(pb, 'child'):
			pb.child.comment = str(rel_path)
			pb.child.wait_for = min(pb.child.wait_for, 10)
			pb.main_bar.comment = str(rel_path)
		else:
			pb.wait_for = min(pb.wait_for, 10)
		pb.comment = "/".join(str(rel_path).split("/")[1:])
		

		if i % 1000 == 0: 
			print(f"Done {i:,d}/{len(df):,d}", flush=True)
			feature_cache.clear()
			synthesis_cache.clear()
			gc.collect()
			time.sleep(4)




def process_weight(weight_para, process_type):
	if process_type == "sum_to_1_geq":
		return F.softmax(weight_para, dim = 1)
		
	elif process_type == "sum_to_1":
		temp = weight_para + 1/weight_para.shape[1]
		return temp/(torch.sum(temp, dim = 1, keepdim = True) + 1e-5)
		
	elif process_type == "geq":
		# so that init close to 1
		# return torch.exp(weight_para)/weight_para.shape[1]
		# return 2*(torch.tanh(weight_para) + 1)/weight_para.shape[1]
		exp_component = torch.exp(weight_para)
		return 3*exp_component/(exp_component + 2)/weight_para.shape[1]
		
	elif process_type == "none":
		# so that init close to 1
		return weight_para + 1/weight_para.shape[1]
		
	else:
		raise NotImplementedError

# ys list of y sequences
def plot_multi_sequences(x, ys, y_names, title = "", template="plotly", width = None, height = None, x_axis = None, y_axis = None, initial_visibility = True, fig = None, fig_row = 1, fig_col = 1):
	'''
	
	import pandas as pd
	data_df = pd.DataFrame(ys, index=y_names, columns=x).T
	
	import plotly.express as px
	# print(data_df)
	fig = px.line(data_df)

	'''
	
	import plotly.graph_objects as go

	# https://community.plotly.com/t/hovertemplate-does-not-show-name-property/36139/2
	
	show_on_screen = (fig is None)
	if show_on_screen:
		fig = go.Figure(data = [go.Scatter(x = x, y = ys[i], name = y_names[i], meta = [y_names[i]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>') for i in range(len(ys))])
	else:
		fig.append_trace(go.Scatter(x = x, y = ys[0], name = y_names[0], meta = [y_names[0]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>'), row=fig_row, col=fig_col)
	
	
	fig.update_layout(
		title=title,
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		margin={"l":40, "r":40, "t":40, "b":40},
		autosize=True,
		template=template,
		width=width,
		height=height,
		xaxis_title=x_axis, 
		yaxis_title=y_axis
	)
	
	
	if not initial_visibility:
		fig.update_traces(visible = 'legendonly')
		
		
	if show_on_screen:
		fig.show(config = {'showTips':False})
	else:
		return fig
	



		
# (batch_size, feature_dim)
def phase_mae(X_1, X_2):	
	# cos_val = F.cosine_similarity(X_1, X_2)
	# multiplier = 1 - cos_val/1.1
	# return torch.mean(multiplier[:, None]*torch.abs(X_1 - X_2), dim = -1)
	
	# print(X_1.shape, X_2.shape)
	# return torch.mean(torch.abs(X_1 - X_2), dim = -1)
	
	return 10**3*torch.mean(F.mse_loss(X_1, X_2, reduction='none'), dim = -1)
	# return 10**7*torch.mean(F.mse_loss(X_1, X_2, reduction='none'), dim = -1)

def wavlm_phase_mae(X_1, X_2):
	return 0.1*torch.mean(F.mse_loss(X_1, X_2, reduction='none'), dim = -1)
	# return torch.mean(F.mse_loss(X_1, X_2, reduction='none'), dim = -1)


def compute_weight(target_feature_indices, synth_set, process_type = "sum_to_1_geq"):
	conv_range = 1
	
	# print(target_feature_indices.shape, synth_set.shape)
	# import sys
	# sys.exit()
	
	
	shape_0, shape_1 = target_feature_indices.shape
	target_feature_indices_surrounding = dict()
	out_feats_surrounding = dict()
	
	for i in range(-conv_range, conv_range + 1):
		target_feature_indices_surrounding[i] = target_feature_indices + i
		
		# avoid dropping out of [0, len(synth_set))
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] < 0] = 0
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] >= len(synth_set)] = len(synth_set) - 1
		
				
		out_feats_surrounding[i] = synth_set[target_feature_indices_surrounding[i].reshape(-1)].reshape(shape_0, shape_1, synth_set.shape[-1])


	weight_para = torch.zeros(target_feature_indices.shape, dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
	import torch.optim as optim
	optimizer = optim.Adam(
		[weight_para], lr = 1e-1, 
		betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
	)

	# process_type = "geq"
	min_loss = 20000
	converge_min_loss = 20000
	import copy
	best_weight_para = copy.deepcopy(weight_para.detach())
	is_loss_decreasing = [True for xxx in range(1000)]
	
	expected_ones = dict()
	
	for t in range(100000):
		
	
		similarities = 0
		for i in range(-conv_range, conv_range+1):
			expected_ones[i] = torch.sum(out_feats_surrounding[i]*process_weight(weight_para, process_type)[..., None], dim = 1)
		
		# separate as we need expected_ones[0]
		for i in range(-conv_range, conv_range+1):
			# print(i, expected_ones[i].shape, expected_ones[0].shape)
			# print(expected_ones[i][-i:].shape, expected_ones[0][:i].shape)
			if i < 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[i][-i:], expected_ones[0][:i]))
				similarity_item = phase_mae(expected_ones[i][-i:], expected_ones[0][:i])
				
				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)
				
			elif i > 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[0][i:], expected_ones[i][:-i]))
				similarity_item = phase_mae(expected_ones[0][i:], expected_ones[i][:-i])

				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)


		loss = similarities
		if t == 0:
			initial_loss = loss.item()
			print(process_weight(weight_para, process_type)[0])
			
		if t % 100 == 1:
			if abs(min_loss - converge_min_loss) < 1e-5:
			# if abs(min_loss - converge_min_loss) < 1e-6:
				break
			
			converge_min_loss = min_loss
			
			
		# , prediction, loss
		if loss < min_loss:
			# if torch.abs(loss - min_loss) > 0.01:
			min_loss = loss.item()
			best_weight_para = copy.deepcopy(weight_para.detach())
			is_loss_decreasing = is_loss_decreasing[1:] + [True]
		else:
			is_loss_decreasing = is_loss_decreasing[1:] + [False]
		
		# break when consecutive 1000 epoch gives no improvement
		if not any(is_loss_decreasing):
			break
		
		# print(weight_para)
		print(t, round(loss.item(), 6), round(initial_loss, 6), end = "\r")
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	# to preserve the last loss print
	print()
	print(process_weight(best_weight_para, process_type)[0])
	
	# sum_to_1_geq
	assert torch.all(process_weight(best_weight_para, process_type) <= 1) and torch.all(process_weight(best_weight_para, process_type) >= 0)
	
	return process_weight(best_weight_para, process_type)
	# return best_weight_para
	



def compute_wavlm_weight(target_feature_indices, synth_set, process_type = "sum_to_1_geq"):
	conv_range = 1
	
	# print(target_feature_indices.shape, synth_set.shape)
	# import sys
	# sys.exit()
	
	
	shape_0, shape_1 = target_feature_indices.shape
	target_feature_indices_surrounding = dict()
	out_feats_surrounding = dict()
	
	for i in range(-conv_range, conv_range + 1):
		target_feature_indices_surrounding[i] = target_feature_indices + i
		
		# avoid dropping out of [0, len(synth_set))
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] < 0] = 0
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] >= len(synth_set)] = len(synth_set) - 1
		
				
		out_feats_surrounding[i] = synth_set[target_feature_indices_surrounding[i].reshape(-1)].reshape(shape_0, shape_1, synth_set.shape[-1])


	weight_para = torch.zeros(target_feature_indices.shape, dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
	import torch.optim as optim
	optimizer = optim.Adam(
		[weight_para], lr = 1e-1, 
		betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
	)

	# process_type = "geq"
	min_loss = 20000
	converge_min_loss = 20000
	import copy
	best_weight_para = copy.deepcopy(weight_para.detach())
	is_loss_decreasing = [True for xxx in range(1000)]
	
	expected_ones = dict()
	
	for t in range(100000):
		
	
		similarities = 0
		for i in range(-conv_range, conv_range+1):
			expected_ones[i] = torch.sum(out_feats_surrounding[i]*process_weight(weight_para, process_type)[..., None], dim = 1)
		
		# separate as we need expected_ones[0]
		for i in range(-conv_range, conv_range+1):
			# print(i, expected_ones[i].shape, expected_ones[0].shape)
			# print(expected_ones[i][-i:].shape, expected_ones[0][:i].shape)
			if i < 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[i][-i:], expected_ones[0][:i]))
				similarity_item = wavlm_phase_mae(expected_ones[i][-i:], expected_ones[0][:i])
				
				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)
				
			elif i > 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[0][i:], expected_ones[i][:-i]))
				similarity_item = wavlm_phase_mae(expected_ones[0][i:], expected_ones[i][:-i])

				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)


		loss = similarities
		if t == 0:
			initial_loss = loss.item()
			print(process_weight(weight_para, process_type)[0])
			
		if t % 100 == 1:
			if abs(min_loss - converge_min_loss) < 1e-5:
			# if abs(min_loss - converge_min_loss) < 1e-6:
				break
			
			converge_min_loss = min_loss
			
			
		# , prediction, loss
		if loss < min_loss:
			# if torch.abs(loss - min_loss) > 0.01:
			min_loss = loss.item()
			best_weight_para = copy.deepcopy(weight_para.detach())
			is_loss_decreasing = is_loss_decreasing[1:] + [True]
		else:
			is_loss_decreasing = is_loss_decreasing[1:] + [False]
		
		# break when consecutive 1000 epoch gives no improvement
		if not any(is_loss_decreasing):
			break
		
		# print(weight_para)
		print(t, round(loss.item(), 6), round(initial_loss, 6), end = "\r")
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	# to preserve the last loss print
	print()
	print(process_weight(best_weight_para, process_type)[0])
	
	# sum_to_1_geq
	assert torch.all(process_weight(best_weight_para, process_type) <= 1) and torch.all(process_weight(best_weight_para, process_type) >= 0)
	
	return process_weight(best_weight_para, process_type)
	# return best_weight_para



def compute_extended_weight(target_feature_indices, synth_set, process_type = "sum_to_1_geq", factors = [1]):
	conv_range = 1
	
	# print(target_feature_indices.shape, synth_set.shape)
	# import sys
	# sys.exit()
	
	
	shape_0, shape_1 = target_feature_indices.shape
	target_feature_indices_surrounding = dict()
	out_feats_surrounding = dict()
	
	for i in range(-conv_range, conv_range + 1):
		target_feature_indices_surrounding[i] = target_feature_indices + i
		
		# avoid dropping out of [0, len(synth_set))
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] < 0] = 0
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] >= len(synth_set)] = len(synth_set) - 1
		
				
		out_feats_surrounding[i] = synth_set[target_feature_indices_surrounding[i].reshape(-1)].reshape(shape_0, shape_1, synth_set.shape[-1])

		out_feats_surrounding[i] = torch.cat([factor*out_feats_surrounding[i] for factor in factors], dim = 1)

	# weight_para = torch.zeros(target_feature_indices.shape, dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
		
	scaling_factors = torch.zeros((target_feature_indices.shape[0], target_feature_indices.shape[1]*len(factors)), dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
	# scaling_max = 1.1
	# scaling_min = 1/1.1
	scaling_max = 1
	scaling_min = 1
		
	weight_para = torch.zeros((target_feature_indices.shape[0], target_feature_indices.shape[1]*len(factors)), dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
	
	import torch.optim as optim
	optimizer = optim.Adam(
		[weight_para, scaling_factors], lr = 1e-1, 
		betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
	)

	# process_type = "geq"
	min_loss = 20000
	converge_min_loss = 20000
	import copy
	best_weight_para = copy.deepcopy(weight_para.detach())
	is_loss_decreasing = [True for xxx in range(1000)]
	
	expected_ones = dict()
	
	for t in range(100000):
		
		
		
		similarities = 0
		for i in range(-conv_range, conv_range+1):
			expected_ones[i] = torch.sum(out_feats_surrounding[i]*process_weight(weight_para, process_type)[..., None]*(torch.tanh(scaling_factors)*(scaling_max - scaling_min)/2 + (scaling_max + scaling_min)/2)[..., None], dim = 1)
		
		# separate as we need expected_ones[0]
		for i in range(-conv_range, conv_range+1):
			# print(i, expected_ones[i].shape, expected_ones[0].shape)
			# print(expected_ones[i][-i:].shape, expected_ones[0][:i].shape)
			if i < 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[i][-i:], expected_ones[0][:i]))
				similarity_item = phase_mae(expected_ones[i][-i:], expected_ones[0][:i])
				
				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)
				
			elif i > 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[0][i:], expected_ones[i][:-i]))
				similarity_item = phase_mae(expected_ones[0][i:], expected_ones[i][:-i])

				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)


		loss = similarities
		if t == 0:
			initial_loss = loss.item()
			print(process_weight(weight_para, process_type)[0])
			
		if t % 100 == 1:
			if abs(min_loss - converge_min_loss) < 1e-5:
			# if abs(min_loss - converge_min_loss) < 1e-6:
				break
			
			converge_min_loss = min_loss
			
			
		# , prediction, loss
		if loss < min_loss:
			# if torch.abs(loss - min_loss) > 0.01:
			min_loss = loss.item()
			best_weight_para = copy.deepcopy(weight_para.detach())
			best_scaling_factors = copy.deepcopy(scaling_factors.detach())
			is_loss_decreasing = is_loss_decreasing[1:] + [True]
		else:
			is_loss_decreasing = is_loss_decreasing[1:] + [False]
		
		# break when consecutive 1000 epoch gives no improvement
		if not any(is_loss_decreasing):
			break
		
		# print(weight_para)
		print(t, round(loss.item(), 6), round(initial_loss, 6), end = "\r")
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	# to preserve the last loss print
	print()
	print(process_weight(best_weight_para, process_type)[0])
	
	# sum_to_1_geq
	assert torch.all(process_weight(best_weight_para, process_type) <= 1) and torch.all(process_weight(best_weight_para, process_type) >= 0)
	
	return process_weight(best_weight_para, process_type)*(torch.tanh(best_scaling_factors)*(scaling_max - scaling_min)/2 + (scaling_max + scaling_min)/2)
	# return best_weight_para




def compute_shift(query_f0, f0_list, target_feature_indices):
	
	shape_0, shape_1 = target_feature_indices.shape
	
	target_feature_f0 = f0_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device)
	median_target_feature_f0 = torch.median(target_feature_f0, dim = -1).values
	
	assert len(query_f0) == len(median_target_feature_f0), [query_f0.shape, median_target_feature_f0.shape]
	
	import copy
	query_f0 = copy.deepcopy(query_f0)
	# inconsistency between pitch extractor and wavlm
	query_f0[median_target_feature_f0 == 0] = 0

	optimal_shift = torch.linalg.lstsq(query_f0.to(median_target_feature_f0.device)[:, None], median_target_feature_f0[:, None]).solution
	
	assert optimal_shift.shape[0] == 1 and optimal_shift.shape[1] == 1
	# print(optimal_shift)
	# import sys
	# sys.exit()
	
	return optimal_shift[0][0]




def compute_weight_with_amp(target_feature_indices, synth_set, process_type = "sum_to_1_geq", amp_ratio = None):
	
	if amp_ratio is not None:
		assert amp_ratio.shape == target_feature_indices.shape
	else:
		# default, nothing changes on amplitude
		amp_ratio = torch.ones_like(target_feature_indices)
	
	
	conv_range = 1
	
	# print(target_feature_indices.shape, synth_set.shape)
	# import sys
	# sys.exit()
	
	
	shape_0, shape_1 = target_feature_indices.shape
	target_feature_indices_surrounding = dict()
	out_feats_surrounding = dict()
	
	for i in range(-conv_range, conv_range + 1):
		target_feature_indices_surrounding[i] = target_feature_indices + i
		
		# avoid dropping out of [0, len(synth_set))
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] < 0] = 0
		target_feature_indices_surrounding[i][target_feature_indices_surrounding[i] >= len(synth_set)] = len(synth_set) - 1
		
				
		out_feats_surrounding[i] = synth_set[target_feature_indices_surrounding[i].reshape(-1)].reshape(shape_0, shape_1, synth_set.shape[-1])*amp_ratio[:, :, None]


	# print(out_feats_surrounding[0].norm(dim=-1, p=1)[250:260])
	# import sys
	# sys.exit()
	

	weight_para = torch.zeros(target_feature_indices.shape, dtype=torch.float32, requires_grad=True, device = target_feature_indices.device)
	import torch.optim as optim
	optimizer = optim.Adam(
		[weight_para], lr = 1e-1, 
		betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True
	)

	# process_type = "geq"
	min_loss = 20000
	converge_min_loss = 20000
	import copy
	best_weight_para = copy.deepcopy(weight_para.detach())
	is_loss_decreasing = [True for xxx in range(1000)]
	
	expected_ones = dict()
	
	for t in range(100000):
		
	
		similarities = 0
		for i in range(-conv_range, conv_range+1):
			expected_ones[i] = torch.sum(out_feats_surrounding[i]*process_weight(weight_para, process_type)[..., None], dim = 1)
		
		# separate as we need expected_ones[0]
		for i in range(-conv_range, conv_range+1):
			# print(i, expected_ones[i].shape, expected_ones[0].shape)
			# print(expected_ones[i][-i:].shape, expected_ones[0][:i].shape)
			if i < 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[i][-i:], expected_ones[0][:i]))
				similarity_item = phase_mae(expected_ones[i][-i:], expected_ones[0][:i])
				
				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)
				
			elif i > 0:
				# similarity_item = (1 - F.cosine_similarity(expected_ones[0][i:], expected_ones[i][:-i]))
				similarity_item = phase_mae(expected_ones[0][i:], expected_ones[i][:-i])

				assert len(similarity_item) == len(target_feature_indices) - abs(i)
				similarities += (1/abs(i))*torch.mean(similarity_item)


		loss = similarities
		if t == 0:
			initial_loss = loss.item()
			print(process_weight(weight_para, process_type)[0])
			
		if t % 100 == 1:
			if abs(min_loss - converge_min_loss) < 1e-5:
			# if abs(min_loss - converge_min_loss) < 1e-6:
				break
			
			converge_min_loss = min_loss
			
			
		# , prediction, loss
		if loss < min_loss:
			# if torch.abs(loss - min_loss) > 0.01:
			min_loss = loss.item()
			best_weight_para = copy.deepcopy(weight_para.detach())
			is_loss_decreasing = is_loss_decreasing[1:] + [True]
		else:
			is_loss_decreasing = is_loss_decreasing[1:] + [False]
		
		# break when consecutive 1000 epoch gives no improvement
		if not any(is_loss_decreasing):
			break
		
		# print(weight_para)
		print(t, round(loss.item(), 6), round(initial_loss, 6), end = "\r")
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	# to preserve the last loss print
	print()
	print(process_weight(best_weight_para, process_type)[0])
	
	# sum_to_1_geq
	assert torch.all(process_weight(best_weight_para, process_type) <= 1) and torch.all(process_weight(best_weight_para, process_type) >= 0)
	
	return process_weight(best_weight_para, process_type)
	# return best_weight_para
	
	

	
	
	
	

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


# per_spk_extract: load all spk folder, for each spk, retain wavlm features and self nearest nbr (set diagonal distance to large ones after computation)
# per_pair_spk_extract: load all spk_folder, form pairs, for each pair, retain wavlm features for each other, and nearest nbr with respect to one another (if self-self then again set diagonal distance to large ones after computation)

# , amp_ratio = None
def sort_by_f0_compatibility(expected_f0, f0_list, target_feature_indices):
	assert len(expected_f0) == len(target_feature_indices)
	
	expected_f0 = expected_f0.to(target_feature_indices.device)
	
	# min_acceptable_f0 = expected_f0*0.97
	# max_acceptable_f0 = expected_f0*1.03
	
	shape_0, shape_1 = target_feature_indices.shape
	# use cpu in case too large
	target_feature_f0 = f0_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device)
	
	# is_good_candidate = (target_feature_f0 >= min_acceptable_f0[:, None])*(target_feature_f0 <= max_acceptable_f0[:, None])
	# https://stackoverflow.com/questions/56088189/pytorch-how-can-i-find-indices-of-first-nonzero-element-in-each-row-of-a-2d-ten
	# is_good_candidate = is_good_candidate*torch.arange(is_good_candidate.shape[1], 0, -1, device = is_good_candidate.device)
	
	# is_good_candidate = torch.abs(target_feature_f0 - expected_f0[:, None])
	is_good_candidate = torch.abs(torch.log2(target_feature_f0 + 1e-5) - torch.log2(expected_f0[:, None] + 1e-5))
	
	
	
	# within the range is sufficient -> 0
	# not within the range -> 1
	# is_good_candidate = 1 - ((target_feature_f0 <= expected_f0[:, None]*2**(1/12) + 1e-5)*(target_feature_f0 >= expected_f0[:, None]/2**(1/12) - 1e-5)).to(int)
	# print(is_good_candidate[200:205, :4])
	# import sys
	# sys.exit()
	
	
	
	'''
	if amp_ratio is not None:
		is_good_candidate[amp_ratio > 10**2] = ((is_good_candidate + 10000)*amp_ratio)[amp_ratio > 10**2]
		is_good_candidate[amp_ratio < 10**(-2)] = ((is_good_candidate + 10000)/amp_ratio)[amp_ratio < 10**(-2)]
	'''
	
		
	# invert a permutation
	# new_target_feature_indices = torch.zeros_like(target_feature_indices)
	# new_target_feature_indices.scatter_(1, is_good_candidate.topk(k=is_good_candidate.shape[1], dim=-1, largest=True).indices, target_feature_indices)
	
	# apply a permutation
	# new_target_feature_indices = target_feature_indices.gather(dim=1, index=is_good_candidate.topk(k=is_good_candidate.shape[1], dim=-1, largest=True).indices)
	new_target_feature_indices = target_feature_indices.gather(dim=1, index=torch.sort(is_good_candidate, dim=1, descending=False, stable=True).indices)
	
	
	'''
	# test_slice = slice(250, 255)
	test_slice = slice(200, 205)
	print(target_feature_indices[test_slice])
	print(f0_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1)[test_slice])
	
	print(new_target_feature_indices[test_slice])
	print(f0_list[new_target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1)[test_slice])
	
	# print(min_acceptable_f0[250:255], max_acceptable_f0[250:255])
	print(expected_f0[test_slice])
	
	import sys
	sys.exit()
	'''
	
	return new_target_feature_indices



# along dim 1
def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
	"""One-dimensional linear interpolation for monotonically increasing sample
	points.

	Returns the one-dimensional piecewise linear interpolant to a function with
	given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

	Args:
		x: the :math:`x`-coordinates at which to evaluate the interpolated
			values.
		xp: the :math:`x`-coordinates of the data points, must be increasing.
		fp: the :math:`y`-coordinates of the data points, same length as `xp`.

	Returns:
		the interpolated values, same size as `x`.
	"""
	import torch
	from torch import Tensor

	# fp: B, F, N, 
	# xp: B, F
	# x: B, 1
	assert len(fp.shape) == 3 and len(xp.shape) == 2 and xp.shape == fp.shape[:2] and x.shape == (xp.shape[0],)
	x = x[:, None]

	x = torch.log(x + 1e-5)
	xp = torch.log(xp + 1e-5)

	# m, b [B, F, N]
	m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:, None] - xp[:, :-1, None])
	b = fp[:, :-1] - (m * xp[:, :-1, None])

	# indices [B, 1] -> [B, F, N]
	indices = torch.sum(torch.ge(x[:, None], xp[:, None, :]), -1) - 1
	# smallest -1 -> 0, biggest F - 1
	indices[indices == -1] = 0
	indices = indices.repeat(1, m.shape[-2], m.shape[-1])
	# indicies = torch.clamp(indicies, 0, m.shape[1] - 1)

	# return m[indicies] * x + b[indicies]
	# [B, 1, N]
	return torch.gather(m, dim = 1, index = indices)[:, :1, :] * x[..., None] + torch.gather(b, dim = 1, index = indices)[:, :1, :]





# matching_type: "wavlm_only", "spec_only", "mix"
# src_wavs, tgt_wavs: generate pool for both, use existing pool if exists. Then for each item in src_wav, output a tuple of needed features (each of len in_wavs) and the pool themselves


def batch_load_wavlm_related(src_wav_files, ref_wav_files, wavlm, selected_layer = 6, topk: int = 4, device = "cuda", prioritize_f0 = False, ckpt_type = "wavlm_only", cache_dir = None) -> Tensor: 
	from pathlib import Path
	src_dataset = set([Path(item).parent for item in src_wav_files])
	ref_dataset = set([Path(item).parent for item in ref_wav_files])
	
	assert len(src_dataset) == 1 and len(ref_dataset) == 1
	

	
	# split into src_wav_file and tgt_wav_file because the save in two types are different, we do not need synth, audio_synth, harmonics_synth in src_wav_files.
	# also, we can just cat everything in ref together as one singer pool
	
	src_query_pools = dict()
	src_spec_pools = dict()
	src_f0_pools = dict()
	
	for wav_file in src_wav_files:
	
		if cache_dir is not None:
			from pathlib import Path
			Path(cache_dir).mkdir(parents = True, exist_ok = True)
		
			import pickle
			cache_pickle = os.path.join(cache_dir, os.path.basename(wav_file) + "_wavlm.npy")
			if os.path.isfile(cache_pickle):
				with open(cache_pickle, "rb") as handle:
					matching_pool, f0_pool = pickle.load(handle)
					
					print("Loaded from", cache_pickle)
			
			else:
			
				matching_pool, _, _, _, f0_pool, _  = get_complete_spk_pool(wav_file, wavlm, match_weights, synth_weights, dataset_root_dir = dataset_root_dir, device = device)
			
				with open(cache_pickle, "wb") as handle:
					pickle.dump((matching_pool, f0_pool), handle)
			
		else:
			
			matching_pool, _, _, _, f0_pool, _  = get_complete_spk_pool(wav_file, wavlm, match_weights, synth_weights, dataset_root_dir = dataset_root_dir, device = device)	
		
		
		src_matching_pools[wav_file] = matching_pool
		src_f0_pools[wav_file] = f0_pool




	tgt_matching_pools = []
	tgt_f0_pools = []
	
	tgt_synth_pools = []
	tgt_audio_synth_pools = []
	tgt_spec_synth_pools = []
	tgt_harmonics_synth_pools = []


	for wav_file in tgt_wav_files:
	
		if cache_dir is not None:
			from pathlib import Path
			Path(cache_dir).mkdir(parents = True, exist_ok = True)
		
			import pickle
			cache_pickle = os.path.join(cache_dir, os.path.basename(wav_file) + "_wavlm.npy")
			if os.path.isfile(cache_pickle):
				with open(cache_pickle, "rb") as handle:
					matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool = pickle.load(handle)
					
					print("Loaded from", cache_pickle)
			
			else:
			
				matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool  = get_complete_spk_pool(wav_file, wavlm, match_weights, synth_weights, dataset_root_dir = dataset_root_dir, device = device)
			
				with open(cache_pickle, "wb") as handle:
					pickle.dump((matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool), handle)
			
		else:
			
			matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool  = get_complete_spk_pool(wav_file, wavlm, match_weights, synth_weights, dataset_root_dir = dataset_root_dir, device = device)	
		
		
		tgt_matching_pools.append(matching_pool)
		tgt_f0_pools.append(f0_pool)
	
		tgt_synth_pools.append(synth_pool)
		tgt_audio_synth_pools.append(audio_synth_pool)
		tgt_spec_synth_pools.append(spec_synth_pool)
		tgt_harmonics_synth_pools.append(harmonics_synth_pool)

	matching_list = torch.concat(tgt_matching_pools, dim=0)
	matching_f0 = torch.concat(tgt_f0_pools, dim=0)
	
	synth_list = torch.concat(tgt_synth_pools, dim=0)
	audio_synth_list = torch.concat(tgt_audio_synth_pools, dim=0)
	spec_synth_list = torch.concat(tgt_spec_synth_pools, dim=0)
	harmonics_synth_list = torch.concat(tgt_harmonics_synth_pools, dim=0)
	


import sys
sys.path.append("/home/ken/open")
from lib_ongaku_test import fast_cosine_dist, knn_cosine_similarity
	


def temp_plot(post_opt, target_feature_indices, synth_feats):
	
	
	
	folder = "/home/ken/Downloads/temp_Choral_not_used/"
	if "no_post_opt" in post_opt:
		src_wav_file = folder + "ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_resampled_16000_knn_mix_no_post_opt.wav"
	else:
		src_wav_file = folder + "ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_resampled_16000_knn_mix_post_opt_0.3.wav"
	
	from lib_ongaku_test import load_audio
	y_1, sr = load_audio(src_wav_file, sr = 16000)
	y_1 = y_1[0]
	print(y_1.shape)
	
	
	start_step_idx = 0
	end_step_idx = 0
	pt_1 = 50.346
	pt_2 = 52.77
	
	ad_1 = 1.22
	ad_2 = 1.56
	
	for time_step_idx in range(len(y_1)):
		if time_step_idx/sr > pt_1 and start_step_idx == 0:
		# if time_step_idx/sr > 34 and start_step_idx == 0:
			start_step_idx = time_step_idx
		elif time_step_idx/sr > pt_2 and end_step_idx == 0:
		# elif time_step_idx/sr > 35 and end_step_idx == 0:
			end_step_idx = time_step_idx
			break

	# pt_1 = 34
	# pt_2 = 35
	# 
	# 
	# print(target_feature_indices[int(pt_1*50):int(pt_2*50)])
	target_feature_chunks = target_feature_indices[int((pt_1 + ad_1)*50):int((pt_1 + ad_2)*50)]
	# print(int((pt_1 + ad_1)*50), int((pt_1 + ad_2)*50), (ad_2 - ad_1)*50)
	# import sys
	# sys.exit()
	import numpy as np
	print(target_feature_chunks.shape, (pt_1 + ad_1 + np.arange(len(target_feature_chunks))/50).shape)
	# import sys
	# sys.exit()
	
	
	
	
	# plot_matrix(target_feature_indices[int(pt_1*50):int(pt_2*50)].T.cpu().numpy(), col_names = pt_1 + np.arange(len(target_feature_indices[int(pt_1*50):int(pt_2*50)]))/50)
	
	# import sys
	# sys.exit()


	# time_slice = slice(start_step_idx, end_step_idx)
	wav_slice = slice(start_step_idx, end_step_idx)
	from scipy.io.wavfile import read
	import matplotlib.pyplot as plt
	import numpy as np
	time = np.linspace(0, len(y_1) / sr, num=len(y_1))
	# plot the first 1024 samples
	# plt.plot(y_1[0, 0:1024])
	# T_1[time_slice], 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
	
	
	from plotly.subplots import make_subplots
	fig = make_subplots(rows=2, cols=1)
	plot_multi_sequences(time[wav_slice], [y_1[wav_slice]], ["w/o CAT"], fig = fig)
	plot_matrix(target_feature_chunks.T.cpu().numpy(), col_names = pt_1 + ad_1 + np.arange(len(target_feature_chunks))/50, fig = fig)
	
	# fig['layout']['xaxis']['title']='Time (s)'
	fig['layout']['xaxis2']['title']='Time (s)'
	fig['layout']['yaxis2']['autorange'] = "reversed"
	print(fig['layout']["height"])
	fig.update_layout(autosize=True, height=450)

	
	fig.write_image(f"/home/ken/Downloads/temp_{post_opt}.pdf")
	# another write to remove the "Loading Mathtype"
	import time
	time.sleep(1.4)
	fig.write_image(f"/home/ken/Downloads/temp_{post_opt}.pdf")
	# fig.show()
	
	import sys
	sys.exit()

	
	




# assume f0 directory's parent is the dataset_root_dir
# any src/ref, for each item in src, convert using the pool of ref
# required_subset used for librispeech only
def match_at_inference_time(src_wav_file, ref_wav_file, wavlm, match_weights, synth_weights, topk: int = 4, device = "cuda", prioritize_f0 = False, ckpt_type = "wavlm_only", src_dataset_path = None, tgt_dataset_path = None, cache_dir = None, required_subset = None, post_opt = "no_post_opt", duration_limit = None) -> Tensor:
	
	# used for f0 cache path locating in get_spk_pool
	if src_dataset_path is None:
		dataset_root_dir = src_wav_file.parent.parent
		assert os.path.isfile(src_wav_file)
	else:
		dataset_root_dir = src_dataset_path
		

	# if duration_limit is not None:
		# to avoid load cache as query
	print("cache dir removed for duration limit")
	cache_dir = None
		
	# cache for dataset (folders of folders) to dataset conversion
	if cache_dir is not None:
		from pathlib import Path
		Path(cache_dir).mkdir(parents = True, exist_ok = True)
		import pickle
		cache_pickle = os.path.join(cache_dir, os.path.basename(src_wav_file) + "_wavlm.npy")
		if os.path.isfile(cache_pickle):
			with open(cache_pickle, "rb") as handle:
				(query_pool, useless_1, useless_2, query_spec_pool, query_f0_pool, useless_3) = pickle.load(handle)
				print("Loaded from", cache_pickle)
			
		else:
			
			query_pool, useless_1, useless_2, query_spec_pool, query_f0_pool, useless_3  = get_complete_spk_pool(src_wav_file, wavlm, match_weights, synth_weights, f0_dir_parent = dataset_root_dir, device = device)
			
			with open(cache_pickle, "wb") as handle:
				pickle.dump((query_pool, useless_1, useless_2, query_spec_pool, query_f0_pool, useless_3), handle)
			
	else:
		
		query_pool, _, _, query_spec_pool, query_f0_pool, _  = get_complete_spk_pool(str(src_wav_file), wavlm, match_weights, synth_weights, f0_dir_parent = dataset_root_dir, device = device)


	if tgt_dataset_path is None:
		dataset_root_dir = ref_wav_file.parent.parent
		assert os.path.isfile(ref_wav_file)
	else:
		dataset_root_dir = tgt_dataset_path

	
	
	if cache_dir is not None:
		from pathlib import Path
		Path(cache_dir).mkdir(parents = True, exist_ok = True)
		import pickle
		cache_pickle = os.path.join(cache_dir, os.path.basename(ref_wav_file) + "_wavlm.npy")
		if os.path.isfile(cache_pickle):
			with open(cache_pickle, "rb") as handle:
				(matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool) = pickle.load(handle)
				print("Loaded from", cache_pickle)
		else:
			# , vad_trigger_level = 7
			matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool = get_complete_spk_pool(str(ref_wav_file), wavlm, match_weights, synth_weights, f0_dir_parent = dataset_root_dir, device = device, duration_limit = duration_limit, vad_trigger_level = 7)

			with open(cache_pickle, "wb") as handle:
				pickle.dump((matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool), handle)
			
	else:
		# , vad_trigger_level = 7
		matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool = get_complete_spk_pool(str(ref_wav_file), wavlm, match_weights, synth_weights, f0_dir_parent = dataset_root_dir, device = device, duration_limit = duration_limit)



	
	matching_list = []
	synth_list = []
	audio_synth_list = []
	spec_synth_list = []
	f0_list = []
	harmonics_synth_list = []
	
	utterance_start_indices = [0]
	
	for item in matching_pool:
		matching_list.append(matching_pool[item])
		synth_list.append(synth_pool[item])
		audio_synth_list.append(audio_synth_pool[item])
		spec_synth_list.append(spec_synth_pool[item])
		f0_list.append(f0_pool[item])
		harmonics_synth_list.append(harmonics_synth_pool[item])
		
		utterance_start_indices.append(utterance_start_indices[-1] + len(matching_pool[item]))
		
		
	matching_list = torch.concat(matching_list, dim=0)
	synth_list = torch.concat(synth_list, dim=0)
	audio_synth_list = torch.concat(audio_synth_list, dim=0)
	spec_synth_list = torch.concat(spec_synth_list, dim=0)
	matching_f0 = torch.concat(f0_list, dim=0)
	harmonics_synth_list = torch.concat(harmonics_synth_list, dim=0)
	
	out_feats_weighted_collection = dict()
	harmonics_out_feats_weighted_collection = dict()
	audio_out_feats_weighted_collection = dict()
	shifted_query_f0_collection = dict()
	
	# assert "1089-134686-0005/1188" in required_subset
	# print("/home/ken/Downloads/knn_vc_data/test/1089/1089-134686-0005.flac" in query_pool.keys())
	# import sys
	# sys.exit()
	
	
	for item in query_pool:
		if required_subset is not None and os.path.basename(item).split(".")[0] + "/" + os.path.basename(ref_wav_file) not in required_subset:
			print(os.path.basename(item).split(".")[0] + "/" + os.path.basename(ref_wav_file))
			# import sys
			# sys.exit()
			continue
	
			
	
		query_seq = query_pool[item]
		query_spec_seq = query_spec_pool[item]
		query_f0 = query_f0_pool[item]
	

	
		matching_start = 0
		increment = 20
		nearest_nbrs_list = []
		
		while matching_start < len(query_seq):
			dists = fast_cosine_dist(query_seq[matching_start:matching_start+increment], matching_list)
		
			
			nearest_nbrs = dists.topk(k=32, dim=-1, largest=False).indices 
			nearest_nbrs_list.append(nearest_nbrs)
				
			matching_start += increment
		
		

		nearest_nbrs = torch.cat(nearest_nbrs_list, dim = 0)
		print(nearest_nbrs.shape)

		


		# self_nearest_nbrs, _ = knn_cosine_similarity(query_seq, query_seq, topk = len(query_seq))
		
		# print(self_nearest_nbrs.shape)
		# import sys
		# sys.exit()
		
		# select f0 based nearest nbrs
		# not possible at training time at mel already fixed
		query_f0_median = torch.median(torch.log(query_f0[query_f0 != 0]))
		matching_f0_median = torch.median(torch.log(matching_f0[matching_f0 != 0]))

		print("query f0 median", torch.exp(query_f0_median), "shape", query_f0.shape)
		print("ref f0 median", torch.exp(matching_f0_median), "shape", matching_f0.shape)


		import copy
		shifted_query_f0 = copy.deepcopy(query_f0)		
		shifted_query_f0[query_f0 != 0] = torch.exp(torch.log(query_f0[query_f0 != 0]) + matching_f0_median - query_f0_median)

		from lib_ongaku_test import smoothen_f0
		
		
		# shifted_query_f0 = torch.tensor(smoothen_f0(shifted_query_f0.cpu().numpy(), [[36.1, 36.16], [51.34, 51.4], [75.14, 75.28], [77.32, 77.52], [157.24,157.34], [193.24, 193.54], [204.1, 204.3], [213.68, 214], [218.9, 219.02], [225.62, 225.78], [226.42, 226.46], [227.16, 227.22], [231.22, 231.32], [231.56, 231.6], [231.8, 231.96]], frame_per_second = 50))
		
		# plot_multi_sequences(np.arange(len(shifted_query_f0))/50, [shifted_query_f0], ["f0"])
		# import sys
		# sys.exit()
	
		import copy
		# target_feature_indices = copy.deepcopy(nearest_nbrs[:, :topk])
		target_feature_indices = copy.deepcopy(nearest_nbrs[:, :4])
		shape_0, shape_1 = target_feature_indices.shape
		
		
		
		print("Indices shape:", target_feature_indices.shape)

		# print(target_feature_indices[200:220])
		# import sys
		# sys.exit()
		def eval_to_wavlm(indices):
			shape_0, shape_1 = indices.shape
			return synth_list[indices.reshape(-1).cpu()].reshape(shape_0, shape_1, synth_list.shape[-1]).to(target_feature_indices.device)
		
		
		
		
		
		# print(eval_to_wavlm(target_feature_indices[1727][None]).shape, eval_to_wavlm(target_feature_indices[1728][None]).shape)
		
		import sys
		sys.path.append("/home/ken/open")
		from lib_ongaku_test import knn_with_concat_cost
		
		# print(target_feature_indices[1727])
		# print(target_feature_indices[1728])
		
		try:
			concat_weight = float(post_opt.split("_")[-1])
		except ValueError:
			if post_opt.split("_")[-1] == "extra":
				concat_weight = 0.3
			else:
				concat_weight = -1
		
		
		query_power_seq = torch.sum(torch.abs(query_spec_seq), dim = -1)
		power_synth_list = torch.sum(torch.abs(spec_synth_list), dim = -1)
		# print(spec_synth_list.shape)
		# import sys
		# sys.exit()
		# plot_matrix(target_feature_indices.T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
		# import sys
		# sys.exit()
		
		
		if concat_weight != -1:
		# if False:
			print(f"using {concat_weight} reselection for wavlm")
			target_feature_indices = knn_with_concat_cost(target_feature_indices, query_seq, matching_list, concat_weight = concat_weight)
		
		# print(target_feature_indices[1727])
		# print(target_feature_indices[1728])
		# plot_multi_sequences(np.arange(len(target_feature_indices))/50, [query_power_seq.cpu().numpy()], ["query"])
		
		# plot_matrix(power_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device).T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50, title = "power")
		
		# plot_matrix(matching_f0[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device).T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50, title = "f0")
		
		
		
		# plot_matrix(target_feature_indices.T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
		
	
		# print(knn_with_concat_cost(target_feature_indices[1727:1729], query_seq[1727:1729],  matching_list))
		# import sys
		# sys.exit()
		
		
		
		
		# print(knn_with_concat_cost(target_feature_indices[1727:1729], query_seq[1727:1729], matching_list))
		# print(target_feature_indices[1727:1729])
		'''
		import sys
		sys.exit()
		
		concat_cost = fast_cosine_dist(eval_to_wavlm(target_feature_indices[1727][None])[0], eval_to_wavlm(target_feature_indices[1728][None])[0])
		
		
		concat_base_cost = fast_cosine_dist(eval_to_wavlm(target_feature_indices[1727][None])[0], eval_to_wavlm(target_feature_indices[1727][None] + 1)[0])
		print(target_feature_indices[1727])
		print(target_feature_indices[1728])
		print(concat_cost)
		print(target_feature_indices[1727]) 
		print(target_feature_indices[1727] + 1)
		print(concat_base_cost)
		
		
		# plot_matrix(concat_cost[1].cpu().numpy(), row_names = target_feature_indices[1727].cpu().numpy(), col_names = target_feature_indices[1728].cpu().numpy())
		# plot_matrix(concat_base_cost[1].cpu().numpy(), row_names = target_feature_indices[1727].cpu().numpy(), col_names = target_feature_indices[1727].cpu().numpy() + 1)
		import sys
		sys.exit()
		'''
		
		'''
		target_feature_indices[1728] = target_feature_indices[1727] + 1
		target_feature_indices[1729] = target_feature_indices[1727] + 2
		target_feature_indices[1730] = target_feature_indices[1727] + 3
		'''
		

		out_feats = synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, synth_list.shape[-1]).to(target_feature_indices.device)
		audio_out_feats = audio_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, audio_synth_list.shape[-1]).to(target_feature_indices.device)
				
		
		
		process_type = "sum_to_1_geq"
		
		# wavlm_weight_para = compute_wavlm_weight(target_feature_indices, synth_list, process_type)
		if "no_post_opt" not in post_opt:
			wavlm_weight_para = compute_wavlm_weight(target_feature_indices, synth_list, process_type)
			out_feats_weighted = torch.sum(out_feats*wavlm_weight_para[..., None], dim = 1).float()
		else:
			
			one_hot_best_weight_para = process_weight(torch.ones_like(out_feats[:, :, 0]), process_type)
			print("Using simple mean", one_hot_best_weight_para)
			
			out_feats_weighted = torch.sum(out_feats*one_hot_best_weight_para[..., None], dim = 1).float()

			
			
		audio_out_feats_weighted = None

		out_feats_weighted_collection[item] = out_feats_weighted
		audio_out_feats_weighted_collection[item] = audio_out_feats_weighted
		shifted_query_f0_collection[item] = shifted_query_f0
		
		
		assert prioritize_f0
		if prioritize_f0:
			nearest_nbrs_f0_priority = sort_by_f0_compatibility(shifted_query_f0, matching_f0, nearest_nbrs)
			
			'''
			plot_matrix(nearest_nbrs[:, :4].T.cpu().numpy(), col_names = np.arange(len(nearest_nbrs_f0_priority))/50)
			plot_matrix(nearest_nbrs_f0_priority[:, :4].T.cpu().numpy(), col_names = np.arange(len(nearest_nbrs_f0_priority))/50)
			import sys
			sys.exit()
			'''
			del nearest_nbrs
			
			
			
			# plot_multi_sequences(np.arange(len(shifted_query_f0))/50, [shifted_query_f0.cpu().numpy()], ["f0"])
			# import numpy as np
			# plot_matrix(nearest_nbrs_f0_priority[:, :8].T.cpu().numpy(), col_names = np.arange(len(nearest_nbrs_f0_priority))/50)
			# import sys
			# sys.exit()
			
			
			
			print("Switching to indices with f0 priority")
			target_feature_indices = copy.deepcopy(nearest_nbrs_f0_priority[:, :4])
			shape_0, shape_1 = target_feature_indices.shape


			# plot_multi_sequences(np.arange(len(target_feature_indices))/50, [shifted_query_f0], ["f0"]) 
			# plot_matrix(target_feature_indices.T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
			# import sys
			# sys.exit()
			# plot_matrix(matching_f0[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device).T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
			
			
			if concat_weight != -1:
				# concat_weight = 0.05
			
				print(f"using {concat_weight} reselection for pitched wavlm")
				#  
				target_feature_indices = knn_with_concat_cost(target_feature_indices, query_seq, matching_list, shifted_query_f0, matching_f0, concat_weight = concat_weight)
				
				
			# temp_plot(post_opt, target_feature_indices, matching_list)
			# import sys
			# sys.exit()	
			
			
			# plot_matrix(power_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device).T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50, title = "power")
			
			# plot_matrix(target_feature_indices.T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
			# plot_matrix(matching_f0[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1).to(target_feature_indices.device).T.cpu().numpy(), col_names = np.arange(len(target_feature_indices))/50)
			# import sys
			# sys.exit()
					
		
		if "wavlm_only" not in ckpt_type and "no_harm_no_amp" not in ckpt_type:

			audio_out_feats = audio_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, audio_synth_list.shape[-1]).to(target_feature_indices.device)
			spec_out_feats = spec_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, spec_synth_list.shape[-1]).to(target_feature_indices.device)
			
			harmonics_out_feats = harmonics_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, harmonics_synth_list.shape[-1]).to(target_feature_indices.device)
			
			if "no_post_opt" not in post_opt:
				process_type = "sum_to_1_geq"
			
				factors = [1]
				harmonics_best_weight_para = compute_extended_weight(target_feature_indices, harmonics_synth_list, process_type, factors)
		

				harmonics_out_feats_weighted = torch.sum(harmonics_out_feats*harmonics_best_weight_para[..., None], dim = 1)
			else:
				harmonics_out_feats_weighted = torch.mean(harmonics_out_feats, dim = 1)
				print("WARNING, using simple mean for harmonics")
		
			
			harmonics_out_feats_weighted_collection[item] = harmonics_out_feats_weighted
	
	
	if "wavlm_only" in ckpt_type or "no_harm_no_amp" in ckpt_type:
		return out_feats_weighted_collection, audio_out_feats_weighted_collection, shifted_query_f0_collection
	elif "mix" in ckpt_type:
		# return torch.cat([out_feats_weighted, spec_out_feats_weighted], dim = 1), audio_out_feats_weighted, shifted_query_f0
		return out_feats_weighted_collection, harmonics_out_feats_weighted_collection, audio_out_feats_weighted_collection, shifted_query_f0_collection
	else:
		raise NotImplementedError

	
	


# @torch.inference_mode()
def per_spk_extract(wavlm: nn.Module, device, ls_path: Path, out_path: Path, synth_weights: Tensor, match_weights: Tensor, save_pool_only = False):
	"""same as extract, but speaker by speaker"""

	# get all folders that contain wav/flac/mp3, take the set of it and these are the spk_folders
	from pathlib import Path
	audio_files = list(Path(ls_path).glob("**/*.wav")) + list(Path(ls_path).glob("**/*.flac"))


	# spk_folder: any audio containing leaf folder under ls_path
	spk_folders = list(set(audio_file.parent for audio_file in audio_files))
	# spk_ids = [str(spk_folder).split("/")[-1] for spk_folder in spk_folders]
	
	# if len(spk_ids) != len(set(spk_ids)):
		# import sys
		# sys.exit("spk_folder names are not unique")
		
		
	# we may not be able to afford loading the entire dataset in at once
	
	# for each pair, for each item in matching_pool, we match it against the entire matching_pool_1 and save the info. Now at runtime, given a file and a target_spk, we also need the target_spk's synth_pool, so save that if not already saved.

	for i in range(len(spk_folders)):
		# if "soprano" not in str(spk_folders[i]):
			# continue
		
		matching_pool, synth_pool, audio_synth_pool, spec_synth_pool, f0_pool, harmonics_synth_pool = get_complete_spk_pool(spk_folders[i], wavlm, match_weights, synth_weights, ls_path, device)
		
		
		
		synth_list = []
		audio_synth_list = []
		spec_synth_list = []
		f0_list = []
		harmonics_synth_list = []
		
		utterance_start_indices = [0]
		
		for item in matching_pool:
			synth_list.append(synth_pool[item])
			audio_synth_list.append(audio_synth_pool[item])
			spec_synth_list.append(spec_synth_pool[item])
			f0_list.append(f0_pool[item])
			harmonics_synth_list.append(harmonics_synth_pool[item])
			
			utterance_start_indices.append(utterance_start_indices[-1] + len(matching_pool[item]))
		
		synth_list = torch.concat(synth_list, dim=0).half().float()
		audio_synth_list = torch.concat(audio_synth_list, dim=0)
		spec_synth_list = torch.concat(spec_synth_list, dim=0)
		f0_list = torch.concat(f0_list, dim=0)
		harmonics_synth_list = torch.concat(harmonics_synth_list, dim=0)
		
		assert utterance_start_indices[-1] == len(synth_list)
		# torch.Size([85042, 320])
		# print(audio_synth_list.shape, synth_list.shape)
		# import sys
		# sys.exit()
		
		# s[i]
		spk_cache_folder = out_path/(spk_folders[i].relative_to(ls_path))
		os.makedirs(spk_cache_folder, exist_ok=True)

		# print(spk_cache_folder)
		# import sys
		# sys.exit()

		import numpy as np
		# if os.path.isfile(str(spk_cache_folder/"pool_spec.npy")):
			# continue
			
		np.save(str(spk_cache_folder/"pool.npy"), synth_list.cpu().numpy())
		# np.save(str(spk_cache_folder/"pool_spec.npy"), spec_synth_list.cpu().numpy())
		np.save(str(spk_cache_folder/"pool_harmonics.npy"), harmonics_synth_list.cpu().numpy())



		# pair may use unsupervised G/D loss, but it may take too much space/complication
		include_cross_nbrs = False
		# now pair folder matching
		if include_cross_nbrs:
			pairing_start = 0
			pairing_end = len(spk_folders)
		else:
			# only the spk and itself
			pairing_start = i
			pairing_end = i + 1



		for j in range(pairing_start, pairing_end):
			
			if j != i:
				assert NotImplementedError
				matching_pool_1, _, _, _, _ = get_complete_spk_pool(spk_folders[j], wavlm, match_weights, synth_weights, device)
			else:
				matching_pool_1 = matching_pool
				# f0_list_1 = f0_list

		
	
			matching_list_1 = []
			for item in matching_pool_1:
				matching_list_1.append(matching_pool_1[item])
			matching_list_1 = torch.concat(matching_list_1, dim=0).half().float()
		
			for k, item in enumerate(matching_pool):
				
				start_index = utterance_start_indices[k]
				end_index = utterance_start_indices[k+1]
			
			
			
				target_feature_path = str(out_path/(Path(item).relative_to(ls_path)).with_suffix('.pt'))
				os.makedirs("/".join(target_feature_path.split("/")[:-1]), exist_ok=True)
		
			
				import pickle
				if os.path.isfile(target_feature_path):
					with open(target_feature_path, 'rb') as handle:
						existing_features = pickle.load(handle)
						
						
					print(existing_features.keys())
					assert existing_features["slice"] == (start_index, end_index)
			
				else:
					existing_features = {"slice": (start_index, end_index)}
				
				with open(target_feature_path, 'wb') as handle:
					pickle.dump(existing_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
				
				if save_pool_only:
					# also save spec for amp_ratio
					np.save(str(spk_cache_folder/"pool_f0.npy"), f0_list.cpu().numpy())
					np.save(str(spk_cache_folder/"pool_spec.npy"), spec_synth_list.cpu().numpy())
					continue			
			
			
				# dists = fast_cosine_dist(matching_list[start_index:end_index], matching_list)
				# print(matching_list_1.shape)
				# import sys
				# sys.exit()
				
				matching_start = 0
				increment = 20
				nearest_nbrs_list = []
				
				while matching_start < len(matching_pool[item]):
					dists = fast_cosine_dist(matching_pool[item][matching_start:matching_start+increment].half().float(), matching_list_1)
						
					# assert dists.shape[0] == len(matching_pool[item])
					# assert dists.shape[0] == increment and dists.shape[1] == matching_list_1.shape[0]
					assert dists.shape[1] == matching_list_1.shape[0]
					
					# print(dists[0, start_index])
					# along the diagonal, set the dist to max
					
					
					if i == j:
						dists[:, start_index:end_index] = 1
					else:
						assert NotImplementedError
					
						
					nearest_nbrs = dists.topk(k=32, dim=-1, largest=False).indices 
					nearest_nbrs_list.append(nearest_nbrs)
						
					matching_start += increment
					
					
				nearest_nbrs = torch.cat(nearest_nbrs_list, dim = 0)
				# print(matching_pool[item].shape, matching_list_1.shape, nearest_nbrs.shape)
				# import sys
				# sys.exit()

				# generate weights
				if i != j:
					import sys
					sys.exit("Potentially bad synth_list, fix that first before cross speaker")
					
				
				nearest_nbrs_f0_priority = sort_by_f0_compatibility(f0_pool[item], f0_list, nearest_nbrs)
				
				import copy
				target_feature_indices = copy.deepcopy(nearest_nbrs[:, :4])
				shape_0, shape_1 = target_feature_indices.shape
				out_feats = synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, synth_list.shape[-1]).to(target_feature_indices.device)
				
				
				# containing negative, therefore sum of vector cannot maintain L1, ignore
			
				# now the one with f0 priority
				target_feature_indices = copy.deepcopy(nearest_nbrs_f0_priority[:, :4])
				
				
								
				
				shape_0, shape_1 = target_feature_indices.shape
				# use cpu in case too large
				audio_out_feats = audio_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, audio_synth_list.shape[-1]).to(target_feature_indices.device)
				spec_out_feats = spec_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, spec_synth_list.shape[-1]).to(target_feature_indices.device)
				
				process_type = "sum_to_1_geq"
				print(item)
				
				
				
				original_L1_norm = spec_synth_pool[item].norm(dim=1, p=1)
				# print(original_L1_norm[250:260])
				knn_L1_norm = spec_out_feats.norm(dim=-1, p=1)
				amp_ratio = original_L1_norm[:, None]/(knn_L1_norm + 1e-5)
				print(torch.max(amp_ratio), torch.min(amp_ratio))
				
				
						
				
				harmonics_best_weight_para = compute_weight_with_amp(target_feature_indices, harmonics_synth_list, process_type, amp_ratio = amp_ratio)
				
				'''
				# harmonics_best_weight_para = compute_weight(target_feature_indices, harmonics_synth_list, process_type)
				'''
				
				
				
				harmonics_out_feats = harmonics_synth_list[target_feature_indices.reshape(-1).cpu()].reshape(shape_0, shape_1, harmonics_synth_list.shape[-1]).to(target_feature_indices.device)
				
				harmonics_out_feats_weighted = torch.sum(harmonics_out_feats*amp_ratio[..., None]*harmonics_best_weight_para[..., None], dim = 1)
		
				
				# harmonics_out_feats_weighted = torch.mean(harmonics_out_feats*amp_ratio[..., None], dim = 1)
				
				# amp_ratio = torch.clamp(amp_ratio, min = 0.2, max = 5)
				
				# harmonics_out_feats_weighted = torch.mean(harmonics_out_feats*amp_ratio[..., None], dim = 1)
				
				# harmonics_out_feats_weighted = harmonics_out_feats[:, 0]*amp_ratio[:, 0, None]
			
				# /50
				# , device = target_feature_indices.device
				'''
				dsp_signal = get_bulk_dsp_choral(f0_pool[item][None, :, None], harmonics_out_feats_weighted[None].to(f0_pool[item]), sample_rate = 16000, hop_size = DOWNSAMPLE_FACTOR, dsp_type = "sin")
				print(torch.max(torch.abs(dsp_signal)), torch.mean(torch.abs(dsp_signal)))
				
				
				# play_sequence(dsp_signal.reshape(-1))
				write_audio("/tmp/temp_3.mp3", dsp_signal.reshape(-1).detach().cpu().numpy(), sample_rate = 16000)
				
				# import sys
				# sys.exit()
				'''
				
				
				'''
				best_weight_para = compute_weight_with_amp(target_feature_indices, spec_synth_list, process_type, amp_ratio = amp_ratio)
								
				one_hot_best_weight_para = F.one_hot(torch.argmax(best_weight_para, dim = 1), num_classes = best_weight_para.shape[1])
				# one_hot_best_weight_para = best_weight_para
				spec_out_feats_weighted = torch.sum(spec_out_feats*amp_ratio[..., None]*one_hot_best_weight_para[..., None], dim = 1)
				audio_out_feats_weighted = torch.sum(audio_out_feats*amp_ratio[..., None]*one_hot_best_weight_para[..., None], dim = 1).reshape(-1)
				
				
				# original_L1_norm = spec_synth_pool[item].norm(dim=1, p=1)
				# knn_weighted_L1_norm = spec_out_feats_weighted.norm(dim=-1, p=1)
				# test_amp_ratio = original_L1_norm/(knn_weighted_L1_norm + 1e-5)
				
				# print(spec_out_feats_weighted.shape, test_amp_ratio[250:310])
				
				write_audio("/tmp/temp_3.mp3", audio_out_feats_weighted.detach().cpu().numpy(), sample_rate = 16000)
				
				# import sys
				# sys.exit()
				'''
			
				
				# import sys
				# sys.exit()
				# print(existing_features.keys())
				# import sys
				# sys.exit()
				
				if include_cross_nbrs:	
					raise NotImplementedError
					# existing_features[spk_ids[j] + "_nearest_nbrs"] = nearest_nbrs.cpu().numpy()
				else:
					# ensure backward compatibility
					existing_features["nearest_nbrs"] = nearest_nbrs.cpu().numpy()
					
					existing_features["nearest_nbrs_f0_priority"] = nearest_nbrs_f0_priority.cpu().numpy()
					# existing_features["best_weights"] = best_weight_para.cpu().numpy()
					
					existing_features["harmonics_best_weight_para"] = harmonics_best_weight_para.cpu().numpy()
					
					if "best_weights" in existing_features:
						del existing_features["best_weights"]
					# existing_features["best_weights"] = (amp_ratio*one_hot_best_weight_para).cpu().numpy()
					
					
					existing_features["amp_ratio"] = amp_ratio.cpu().numpy()
					# if "amp_ratio" in existing_features:
						# del existing_features["amp_ratio"]
					
					# existing_features["f0"] = f0_pool[item]
				
				with open(target_feature_path, 'wb') as handle:
					pickle.dump(existing_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# print(i, "/", len(spk_folders), "/".join(str(spk_folders[i]).split("/")[-3:]), end="\r", flush=True)
		print(i, "/", len(spk_folders), "/".join(str(spk_folders[i]).split("/")[-3:]), flush=True)



def main(args):
	device = torch.device(args.device)
	SYNTH_WEIGHTINGS = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=25).float().to(device)[:, None]
	MATCH_WEIGHTINGS = F.one_hot(torch.tensor(args.matching_layer), num_classes=25).float().to(device)[:, None]

	print(f"Matching weightings: {MATCH_WEIGHTINGS.squeeze()}\nSynthesis weightings: {SYNTH_WEIGHTINGS.squeeze()}")
	# get all files and their corresponding speakers
	# ls_df = make_librispeech_df(Path(args.librispeech_path))

	print(f"Loading wavlm.")
	wavlm = wavlm_large(pretrained=True, progress=True, device=args.device)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# extract(ls_df, wavlm, args.device, Path(args.librispeech_path), Path(args.out_path), SYNTH_WEIGHTINGS, MATCH_WEIGHTINGS)
	
	per_spk_extract(wavlm, args.device, Path(args.librispeech_path), Path(args.out_path), SYNTH_WEIGHTINGS, MATCH_WEIGHTINGS, save_pool_only = False)
	print("All done!", flush=True)






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Compute matched wavlm features for a librispeech dataset")

	parser.add_argument('--librispeech_path', required=True, type=str)
	parser.add_argument('--seed', default=123, type=int)
	parser.add_argument('--out_path', required=True, type=str)
	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--topk', type=int, default=4)
	parser.add_argument('--matching_layer', type=int, default=6)
	parser.add_argument('--synthesis_layer', type=int, default=6)
	parser.add_argument('--prematch', action='store_true', help='prematch')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--include_cross_nbrs', type=bool, default=False)

	args = parser.parse_args()
	main(args)

# python prematch_dataset.py --librispeech_path /home/ken/Downloads/transplayer_data/train --out_path /home/ken/Downloads/transplayer_data/cached/train --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch --include_cross_nbrs True
# python prematch_dataset.py --librispeech_path /home/ken/Downloads/transplayer_data/valid --out_path /home/ken/Downloads/transplayer_data/cached/valid --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch --include_cross_nbrs True

# python ddsp_prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/train/ --out_path /home/ken/Downloads/knn_vc_data/cached/train/ --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch
# python ddsp_prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/Cantoria/valid/ --out_path /home/ken/Downloads/knn_vc_data/Cantoria_cached/valid/ --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch


# python ddsp_prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/train/ --out_path /home/ken/Downloads/knn_vc_data/cached/train/ --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch; python ddsp_prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/OpenSinger_train/ --out_path /home/ken/Downloads/knn_vc_data/cached/OpenSinger_train/ --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch

# python ddsp_prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/OpenSinger_train/ --out_path /home/ken/Downloads/knn_vc_data/cached/OpenSinger_train/ --topk 4 --matching_layer 6 --synthesis_layer 6  --prematch
