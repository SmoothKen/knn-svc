
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor
from wavlm.WavLM import WavLM
from knnvc_utils import generate_matrix_from_index


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
	waveform_abs_max = np.max(np.abs(waveform))
	if waveform_abs_max > 1:
		waveform = waveform/waveform_abs_max
	
	
	# convert to int32 if it is [-1, 1] float
	if waveform.dtype == np.float32 or waveform.dtype == np.float64:
		waveform = waveform * (2 ** 31 - 1)   
		waveform = waveform.astype(np.int32)
	else:
		assert waveform.dtype == np.int32
		
		

	if filename.endswith(".wav"):
		import soundfile as sf
		sf.write(filename, waveform.T, samplerate = sample_rate, subtype = 'PCM_32')
		
	else:
		from pydub import AudioSegment
		audio_segment = AudioSegment(
			waveform.T.tobytes(), 
			frame_rate=sample_rate,
			sample_width=4, 
			channels=waveform.shape[0]
		)
		audio_segment.export(filename, format=filename.split(".")[-1], bitrate="320k")

# ys list of y sequences
def plot_multi_sequences(x, ys, y_names, title = "", template="plotly", width = None, height = None, x_axis = None, y_axis = None, initial_visibility = True):
	'''
	
	import pandas as pd
	data_df = pd.DataFrame(ys, index=y_names, columns=x).T
	
	import plotly.express as px
	# print(data_df)
	fig = px.line(data_df)

	'''
	
	import plotly.graph_objects as go

	# https://community.plotly.com/t/hovertemplate-does-not-show-name-property/36139/2
	if type(x[0]) == list:
		assert len(x) == len(ys)
		fig = go.Figure(data = [go.Scatter(x = x[i], y = ys[i], name = y_names[i], meta = [y_names[i]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>') for i in range(len(ys))])
	else:
		assert all(len(x) == len(ys[i]) for i in range(len(ys)))
		fig = go.Figure(data = [go.Scatter(x = x, y = ys[i], name = y_names[i], meta = [y_names[i]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>') for i in range(len(ys))])
	
	
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
		
	fig.show(config = {'showTips':False})
	
	

# assume col_names is the same as row_names 
def plot_matrix(mat, row_names = None, col_names = None, title = ""):

	import plotly.express as px

	fig = px.imshow(mat, text_auto=True, x=col_names, y=row_names, aspect='auto', color_continuous_scale='Bluered_r')
	fig.update_layout(
		title=title,
		xaxis_title="",
		yaxis_title="",
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		margin={"l":0, "r":0, "t":40, "b":0},
		autosize=True,
		template="simple_white"
	)

	fig.show()	
	


SPEAKER_INFORMATION_LAYER = 6
SPEAKER_INFORMATION_WEIGHTS = generate_matrix_from_index(SPEAKER_INFORMATION_LAYER)
print("weights", SPEAKER_INFORMATION_WEIGHTS)


import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F

class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=1024, alpha=1.0, overlap=0, eps=1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=1, normalized=True, center=False)
        
    def forward(self, x_true, x_pred):
        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps
        
        converge_term = torch.mean(torch.linalg.norm(S_true - S_pred, dim = (1, 2)) / torch.linalg.norm(S_true + S_pred, dim = (1, 2)))
        
        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
        return loss
        
        
class RSSLoss(nn.Module):
    '''
    Random-scale Spectral Loss.
    '''

    def __init__(self, fft_min = 256, fft_max = 2048, n_scale = 4, alpha=1.0, overlap=0, eps=1e-7, device='cuda'):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max
        self.n_scale = n_scale
        self.lossdict = {}
        for n_fft in range(fft_min, fft_max):
            self.lossdict[n_fft] = SSSLoss(n_fft, alpha, overlap, eps).to(device)
        
    def forward(self, x_pred, x_true):
        value = 0.
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            loss_func = self.lossdict[int(n_fft)]
            value += loss_func(x_true, x_pred)
        return value / self.n_scale
            
        
    
    
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
		

'''

def phase_mae(estimated_wave, target_wave):
	# assume 16000, wavlm_hop 320
	# fft 1024, hop 160
	waves = torch.stack([estimated_wave, target_wave], dim = 0)

	X = torch.stft(waves, 1024, 160,
			window=torch.hann_window(1024).to(waves),
			win_length=1024,
			normalized=True,
			center=True,
			return_complex=True,
			pad_mode='reflect')
	
	input = X[0]
	target = X[1]
	
	cos_val = (input.real*target.real + input.imag*target.imag)/((torch.abs(input) + 1e-5)*(torch.abs(target) + 1e-5))
	multiplier = 1 - cos_val/2
	# multiplier = 1.5 - cos_val
	return torch.mean(multiplier*torch.abs(input - target))
'''


# (batch_size, feature_dim)
def phase_mae(X_1, X_2):	
	# cos_val = F.cosine_similarity(X_1, X_2)
	# multiplier = 1 - cos_val/1.1
	# return torch.mean(multiplier[:, None]*torch.abs(X_1 - X_2), dim = -1)
	
	print(X_1.shape, X_2.shape)
	return torch.mean(torch.abs(X_1 - X_2), dim = -1)
	# print(multiplier.shape, X_1.shape)
	# import sys
	# sys.exit()
	


# essentially compute the a \dot b = (-(a - b)**2 + a^2 + b^2)/2, then 1 - a \dot b/||a||||b|| as the final distance
def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
	""" Like torch.cdist, but fixed dim=-1 and for cosine distance."""
	source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
	matching_norms = torch.norm(matching_pool, p=2, dim=-1)
	dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
	dotprod /= 2

	dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
	return dists



def fast_weighted_cosine_dist(source_feats: Tensor, matching_pool: Tensor, weights = None, device: str = 'cpu') -> Tensor:
	""" Like torch.cdist, but fixed dim=-1 and for cosine distance."""
	
	
	if weights is None:
		weights = torch.ones(source_feats.shape, device = source_feats.device)
		
		
	# start with [B_1, M], [B_2, M], weights [B_1, M] 
	
	# print(source_feats.shape, matching_pool.shape)
	# import sys
	# sys.exit()
	
	# for each (i, j) pair, \sum_ w_ix_iy_j/(norm(w_ix_i)*norm(w_iy_j))
	weighted_source_feats = source_feats*weights
	
	source_norms = torch.norm(weighted_source_feats, p=2, dim=-1).to(device)
	
	
	
	
	matching_norms = []
	current_idx = 0
	
	# [B_1, 1, M]*[1, B_2, M] -> [B_1, B_2, M] -- norm --> [B_1, B_2]
	while current_idx < len(weights):
		weighted_matching_pool = weights[current_idx:current_idx+10, None]*matching_pool[None, :]
		matching_norms.append(torch.norm(weighted_matching_pool, p=2, dim=-1))
		
		current_idx += 10
	
	matching_norms = torch.cat(matching_norms, dim = 0)
	# print(matching_norms.shape)
	# import sys
	# sys.exit()
	

	# dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
	# dotprod /= 2
	dotprod = torch.einsum('id,jd->ij', weighted_source_feats, matching_pool)

	# denominator [B_1, 1]*[B_1, B_2]
	dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms) )
	return dists




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




class KNeighborsVC(nn.Module):

	def __init__(self,
		wavlm: WavLM,
		hifigan: HiFiGAN,
		hifigan_cfg: AttrDict,
		device='cuda'
	) -> None:
		""" kNN-VC matcher. 
		Arguments:
			- `wavlm` : trained WavLM model
			- `hifigan`: trained hifigan model
			- `hifigan_cfg`: hifigan config to use for vocoding.
		"""
		super().__init__()
		# set which features to extract from wavlm
		self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
		# load hifigan
		self.hifigan = hifigan.eval()
		self.h = hifigan_cfg
		# store wavlm
		self.wavlm = wavlm.eval()
		self.device = torch.device(device)
		self.sr = self.h.sampling_rate
		self.hop_length = 320

	# : list[Path] | list[Tensor]
	# def get_matching_set(self, wavs, weights=None, vad_trigger_level=7) -> Tensor:
	def get_matching_set(self, wavs, weights=None, vad_trigger_level=7) -> Tensor:
		""" Get concatenated wavlm features for the matching set using all waveforms in `wavs`, 
		specified as either a list of paths or list of loaded waveform tensors of 
		shape (channels, T), assumed to be of 16kHz sample rate.
		Optionally specify custom WavLM feature weighting with `weights`.
		"""
		feats = []
		for p in wavs:
			feats.append(self.get_features(p, weights=self.weighting if weights is None else weights, vad_trigger_level=vad_trigger_level))
		
		feats = torch.concat(feats, dim=0).cpu()
		return feats
		

	@torch.inference_mode()
	def mel_vocode(self, c:Tensor, f0: Tensor):

		c = c.unsqueeze(0)
		
		mel = LogMelSpectrogram(self.h.n_fft, self.h.num_mels, self.h.sampling_rate, self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax)(c.to("cpu")).to("cuda")
		mel = mel.permute(0, 2, 1)
		if len(mel.shape) < 3:
			mel = mel.unsqueeze(0) # (1, seq_len, dim)

		# print(c.shape, mel.shape)
		# import sys
		# sys.exit()


		f0 = f0[:mel.shape[1]].unsqueeze(0).unsqueeze(0).to(self.device)
		
		# print(c.shape, f0.shape)
		# import sys
		# sys.exit()

		y_g_hat = self.hifigan(mel, f0)
		y_g_hat = y_g_hat.squeeze(1)
		return y_g_hat

	# : Tensor
	# (bs, seq_len, c_dim)
	# (bs, seq_len, 1)
	# (bs, seq_len, harmonics)
	@torch.inference_mode()
	def vocode(self, c: Tensor, f0 = None, harmonics_out_feats_weighted = None) -> Tensor:
		""" Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim) """
		
		# [1, 1, seq_len]
		# print(c.shape, f0.shape)
		if f0 is not None:
		
		
			if harmonics_out_feats_weighted is not None:
				print("mix", c.shape, f0.shape, harmonics_out_feats_weighted.shape)
				# import sys
				# sys.exit()
				y_g_hat = self.hifigan(c, f0.to(c), harmonics_out_feats_weighted.to(c))
			else:
				print("wavlm_only", f0.shape)
				# import sys
				# sys.exit()
				y_g_hat = self.hifigan(c, f0)
				
		else:
			y_g_hat = self.hifigan(c)
			
			
		# print(c, y_g_hat)
		# import sys
		# sys.exit()
		# print(y_g_hat.shape)
		# import sys
		# sys.exit()
		
		y_g_hat = y_g_hat.squeeze(1)
		return y_g_hat



	def get_f0(self, wav_file):
		import pyworld as pw
		import librosa
		audio, sr = librosa.load(
			wav_file, 
			sr=None)
			
		assert sr == 16000
		

		import numpy as np
		f0, _ = pw.harvest(audio.astype(np.float64), self.h.sampling_rate, f0_floor=65.0, f0_ceil=1047.0, frame_period=self.h.hop_size / self.h.sampling_rate * 1000)
			
		f0_hz = torch.from_numpy(f0).float()
		f0_hz[f0_hz<80] *= 0
		
		return f0_hz

	def get_multiple_f0(self, wav_files):
		f0_results = []
		for wav_file in wav_files:
			f0_results.append(self.get_f0(wav_file))
			
		return torch.cat(f0_results)


	
	@torch.inference_mode()
	def get_features(self, path, weights=None, vad_trigger_level=0, return_audio = False):
		"""Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
		on start/end with `vad_trigger_level`.
		"""
		# load audio
		if weights == None: weights = self.weighting
		if type(path) in [str, Path]:
			x, sr = torchaudio.load(path, normalize=True)
			if len(x.shape) == 2:
				x = x[0][None, :]
		else:
			x: Tensor = path
			sr = self.sr
			if x.dim() == 1: x = x[None]
				
		if not sr == self.sr :
			print(f"resample {sr} to {self.sr} in {path}")
			x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
			sr = self.sr
			
			
		lstrip_len = 0
		rstrip_len = 0
		# trim silence from front and back
		if vad_trigger_level > 1e-3:
			print("VAD trimming", path, vad_trigger_level)
			transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
			x_front_trim = transform(x)
			lstrip_len = x.shape[-1] - x_front_trim.shape[-1]
			if lstrip_len % self.hop_length != 0:
				extra_cut = self.hop_length - lstrip_len % self.hop_length
				
				x_front_trim = x_front_trim[extra_cut:]
				lstrip_len += extra_cut
				
			# original way, disabled because it lacks windows support
			#waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
			waveform_reversed = torch.flip(x_front_trim, (-1,))
			waveform_reversed_front_trim = transform(waveform_reversed)
			rstrip_len = waveform_reversed.shape[-1] - waveform_reversed_front_trim.shape[-1]
			if rstrip_len % self.hop_length != 0:
				extra_cut = self.hop_length - rstrip_len % self.hop_length
				
				waveform_reversed_front_trim = waveform_reversed_front_trim[extra_cut:]
				rstrip_len += extra_cut
			
			
			
			waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
			#waveform_end_trim, sr = apply_effects_tensor(
			#    waveform_reversed_front_trim, sr, [["reverse"]]
			#)
			
			x = waveform_end_trim

		# extract the representation of each layer
		print(lstrip_len, rstrip_len)
		# import sys
		# sys.exit()
		
		wav_input_16khz = x.to(self.device)
		print("input wav shape", wav_input_16khz.shape)
		
		if torch.allclose(weights, self.weighting):
			# use fastpath
			features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
			features = features.squeeze(0)
			print("WavLM features shape", features.shape)
		else:
			# use slower weighted
			rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
			features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
			# save full sequence
			features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
		

		if return_audio:
			return features, wav_input_16khz
		else:
			return features


	@torch.inference_mode()
	# : float | None  : float | None : str | None
	def match(self, query_seq: Tensor, matching_set: Tensor, query_f0: Tensor, synth_set: Tensor = None, 
			  topk: int = 4, tgt_loudness_db = -16,
			  target_duration = None, device = None, without_vocode = False) -> Tensor:
		""" Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
		with k=`topk`. Inputs:
			- `query_seq`: Tensor (N1, dim) of the input/source query features.
			- `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
			- `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
				vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
				By default, and for best performance, this should be identical to the matching set. 
			- `topk`: k in the kNN -- the number of nearest neighbors to average over.
			- `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable. 
			- `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
			- `device`: if None, uses default device at initialization. Otherwise uses specified device
		Returns:
			- converted waveform of shape (T,)
		"""
		device = torch.device(device) if device is not None else self.device
		if synth_set is None: synth_set = matching_set.to(device)
		else: synth_set = synth_set.to(device)
		matching_set = matching_set.to(device)
		query_seq = query_seq.to(device)

		if target_duration is not None:
			target_samples = int(target_duration*self.sr)
			scale_factor = (target_samples/self.hop_length) / query_seq.shape[0] # n_targ_feats / n_input_feats
			query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

		dists = fast_cosine_dist(query_seq, matching_set, device=device)
		
		
		# dists = fast_cosine_dist(query_seq, query_seq, device=device)
		best = dists.topk(k=topk, largest=False, dim=-1)
		
		
		import numpy as np
		# print(dists.shape, best.indices.shape, dists[best.indices].shape)
		longer_best = dists.topk(k=18, largest=False, dim=-1)
		
		# np.savetxt('/home/ken/Downloads/temp.txt', longer_best.indices.cpu().numpy().astype(int), fmt='%i')
		# using real time isntead
		
		# torch.Size([2785, 13]) torch.Size([2785, 12])
		temp_indices = torch.cat([torch.arange(longer_best.indices.shape[0]).to(longer_best.indices)[:, None], longer_best.indices], dim = 1)
		# print(temp_indices.shape, longer_best.indices.shape)
		# import sys
		# sys.exit()
		
		np.savetxt('/home/ken/Downloads/temp.txt', temp_indices.cpu().numpy().astype(int)*320/16000, fmt='%.3f')
		np.savetxt('/home/ken/Downloads/temp_dist.txt', longer_best.values.cpu().numpy(), fmt="%.3f")
		# plot_matrix(best.indices.cpu().numpy(), row_names = [str(i) for i in range(best.indices.cpu().numpy().shape[0])], col_names = [str(i) for i in range(best.indices.cpu().numpy().shape[1])])
		
		print(best.indices[200:220])
		import sys
		sys.exit()
		
		out_feats = synth_set[best.indices].mean(dim=1)
		# best_weights = F.softmax(-best.values**2, dim = 1)
		# np.savetxt('/home/ken/Downloads/temp_weights.txt', best_weights.cpu().numpy(), fmt="%.3f")
		# out_feats = torch.sum(synth_set[best.indices]*best_weights[..., None], dim = 1)
		
		
		assert out_feats.shape == query_seq.shape
		if without_vocode:
			return out_feats

		'''
		dists_2 = fast_cosine_dist(out_feats, matching_set, device=device)
		longer_best_2 = dists_2.topk(k=12, largest=False, dim=-1)

		import numpy as np

		np.savetxt('/home/ken/Downloads/temp_out.txt', longer_best_2.indices.cpu().numpy().astype(int), fmt='%i')
		np.savetxt('/home/ken/Downloads/temp_out_dist.txt', longer_best_2.values.cpu().numpy(), fmt="%.3f")
		
		'''
		
		
		
		# import sys
		# sys.exit()
		# query_shape [T, F]
		# print(query_seq.shape)
		# import sys
		# sys.exit()

		
		'''
		import numpy as np
		plot_matrix(query_seq.cpu().numpy().T, row_names = np.arange(query_seq.cpu().numpy().T.shape[0]), col_names = np.arange(query_seq.cpu().numpy().T.shape[1])*320/16000)
		plot_matrix(out_feats.cpu().numpy().T, row_names = np.arange(out_feats.cpu().numpy().T.shape[0]), col_names = np.arange(out_feats.cpu().numpy().T.shape[1])*320/16000)
		import sys
		sys.exit()	
		'''
			
		# test original
		# out_feats = query_seq
		
		# plot_matrix(dists.cpu().numpy(), row_names = [str(i) for i in 
		
		print("dists shape", dists.shape, "out_feats shape", out_feats.shape)
		# prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()
		
		prediction = self.vocode(out_feats[None].to(device), query_f0).cpu().squeeze()
		
		print("prediction shape", prediction.shape)
		# normalization
		'''
		if tgt_loudness_db is not None:
			src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
			tgt_loudness = tgt_loudness_db
			pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
		else: pred_wav = prediction
		'''
		pred_wav = prediction
		return pred_wav
		




	@torch.inference_mode()
	# : float | None  : float | None : str | None
	def self_match(self, query_seq: Tensor, matching_set: Tensor, query_f0: Tensor, synth_set: Tensor = None, 
			  topk: int = 4, tgt_loudness_db = -16,
			  target_duration = None, device = None, without_vocode = False) -> Tensor:
		""" Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
		with k=`topk`. Inputs:
			- `query_seq`: Tensor (N1, dim) of the input/source query features.
			- `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
			- `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
				vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
				By default, and for best performance, this should be identical to the matching set. 
			- `topk`: k in the kNN -- the number of nearest neighbors to average over.
			- `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable. 
			- `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
			- `device`: if None, uses default device at initialization. Otherwise uses specified device
		Returns:
			- converted waveform of shape (T,)
		"""
		device = torch.device(device) if device is not None else self.device
		if synth_set is None: synth_set = matching_set.to(device)
		else: synth_set = synth_set.to(device)
		matching_set = matching_set.to(device)
		query_seq = query_seq.to(device)

		if target_duration is not None:
			target_samples = int(target_duration*self.sr)
			scale_factor = (target_samples/self.hop_length) / query_seq.shape[0] # n_targ_feats / n_input_feats
			query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

		dists = fast_cosine_dist(query_seq, matching_set, device=device)
		
		
		# dists = fast_cosine_dist(query_seq, query_seq, device=device)
		best = dists.topk(k=topk, largest=False, dim=-1)
		
		
		import numpy as np
		# print(dists.shape, best.indices.shape, dists[best.indices].shape)
		longer_best = dists.topk(k=18, largest=False, dim=-1)
		
		# np.savetxt('/home/ken/Downloads/temp.txt', longer_best.indices.cpu().numpy().astype(int), fmt='%i')
		# using real time isntead
		
		# torch.Size([2785, 13]) torch.Size([2785, 12])
		temp_indices = torch.cat([torch.arange(longer_best.indices.shape[0]).to(longer_best.indices)[:, None], longer_best.indices], dim = 1)
		# print(temp_indices.shape, longer_best.indices.shape)
		# import sys
		# sys.exit()
		
		np.savetxt('/home/ken/Downloads/temp.txt', temp_indices.cpu().numpy().astype(int)*320/16000, fmt='%.3f')
		np.savetxt('/home/ken/Downloads/temp_dist.txt', longer_best.values.cpu().numpy(), fmt="%.3f")
		# plot_matrix(best.indices.cpu().numpy(), row_names = [str(i) for i in range(best.indices.cpu().numpy().shape[0])], col_names = [str(i) for i in range(best.indices.cpu().numpy().shape[1])])
		
		out_feats = synth_set[best.indices].mean(dim=1)
		# best_weights = F.softmax(-best.values**2, dim = 1)
		# np.savetxt('/home/ken/Downloads/temp_weights.txt', best_weights.cpu().numpy(), fmt="%.3f")
		# out_feats = torch.sum(synth_set[best.indices]*best_weights[..., None], dim = 1)
		
		
		assert out_feats.shape == query_seq.shape
		if without_vocode:
			return out_feats

		'''
		dists_2 = fast_cosine_dist(out_feats, matching_set, device=device)
		longer_best_2 = dists_2.topk(k=12, largest=False, dim=-1)

		import numpy as np

		np.savetxt('/home/ken/Downloads/temp_out.txt', longer_best_2.indices.cpu().numpy().astype(int), fmt='%i')
		np.savetxt('/home/ken/Downloads/temp_out_dist.txt', longer_best_2.values.cpu().numpy(), fmt="%.3f")
		
		'''
		
		
		
		# import sys
		# sys.exit()
		# query_shape [T, F]
		# print(query_seq.shape)
		# import sys
		# sys.exit()

		
		'''
		import numpy as np
		plot_matrix(query_seq.cpu().numpy().T, row_names = np.arange(query_seq.cpu().numpy().T.shape[0]), col_names = np.arange(query_seq.cpu().numpy().T.shape[1])*320/16000)
		plot_matrix(out_feats.cpu().numpy().T, row_names = np.arange(out_feats.cpu().numpy().T.shape[0]), col_names = np.arange(out_feats.cpu().numpy().T.shape[1])*320/16000)
		import sys
		sys.exit()	
		'''
			
		# test original
		# out_feats = query_seq
		
		# plot_matrix(dists.cpu().numpy(), row_names = [str(i) for i in 
		
		print("dists shape", dists.shape, "out_feats shape", out_feats.shape)
		# prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()
		
		# prediction = self.vocode(out_feats[None].to(device), query_f0).cpu().squeeze()
		prediction = self.vocode(query_seq[None], query_f0).cpu().squeeze()
		
		print("prediction shape", prediction.shape)
		# normalization
		'''
		if tgt_loudness_db is not None:
			src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
			tgt_loudness = tgt_loudness_db
			pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
		else: pred_wav = prediction
		'''
		
		pred_wav = prediction
		return pred_wav



	# query_f0, already shifted and should now be comparable to f0
	def select_neighbors(self, best, target_feature_indices, is_going_up, matching_audio, query_f0, f0):

		prediction = []
		prediction_f0 = []
		# print(len(best.indices), len(query_f0))
		if abs(len(best.indices) - len(query_f0)) > 2:
			print("Shape mismatch", len(best.indices), len(query_f0))
			import sys
			sys.exit()
		
		for i in range(len(best.indices)):

			# allow 0, or fall into (min/1.2, max*1.2)

			f0_local_range = query_f0[max(0, i-3):min(i+4, len(query_f0))]
			
				
			print(f0_local_range, end = " ")
			if torch.sum(f0_local_range == 0).item() > 0:
				allow_zero = True
			else:
				allow_zero = False
				
			f0_local_range = f0_local_range[f0_local_range != 0]
			if len(f0_local_range) > 0:
				f0_local_range = (torch.min(f0_local_range)/1.03, torch.max(f0_local_range)*1.03)
			else:
				f0_local_range = (0, 0)
				
			print(round(i*320/16000, 3), allow_zero, f0_local_range)
				
			'''
			if is_going_up:
				f0_local_range = query_f0[max(0, i-3):min(i+4, len(query_f0))]
				if torch.count_nonzero(f0_local_range).item() > 0:
					current_expected_f0 = torch.max(f0_local_range[f0_local_range > 0])
				else:
					# super high number
					current_expected_f0 = torch.zeros(1)
			else:
				f0_local_range = query_f0[max(0, i-3):min(i+4, len(query_f0))]
				if torch.count_nonzero(f0_local_range).item() > 0:
					current_expected_f0 = torch.min(f0_local_range[f0_local_range > 0])
				else:
					current_expected_f0 = torch.zeros(1)
				
			'''
				
			'''
			if i > len(query_f0):
				print(query_f0[max(0, i-3):min(i+4, len(query_f0))],  torch.count_nonzero(f0_local_range).item(), current_expected_f0)
			'''

			chosen_ranks = []
			chosen_candidates = []
			for m, item in enumerate(best.indices[i]):
				
				'''
				if is_going_up:
					if current_expected_f0 == 0:
						pass
						# if f0[item] != 0:
							# continue
					else:
						if f0[item] <= current_expected_f0*0.9:
							continue
				else:
					if current_expected_f0 == 0:
						# whatever
						pass
						# if f0[item] != 0:
							# continue
					else:
						if f0[item] >= current_expected_f0*10/9:
							continue
				
				'''
				
				'''
				if current_expected_f0 == 0:
					# whatever
					pass
					# if f0[item] != 0:
						# continue
				else:
					if f0[item] >= current_expected_f0*1.2 or f0[item] <= current_expected_f0*0.8:
						continue
				'''
				
				if allow_zero and f0[item] == 0:
					pass
				# elif f0[item] >= f0_local_range[0] or f0[item] <= f0_local_range[1]:
				elif f0[item] >= f0_local_range[0] and f0[item] <= f0_local_range[1]:
					pass
				else:
					continue
				
				
				
				'''
				is_bad_choice = False
				for candidate in chosen_candidates:
					if torch.abs(candidate - item) < 30:
						is_bad_choice = True
						break
				
				if is_bad_choice:
					continue
				else:
					chosen_candidates.append(item)
				'''
				# print(m)
				chosen_candidates.append(item)
				chosen_ranks.append(m)

			# print(chosen_ranks[:4])
			# if max(chosen_ranks[:4]) >= 4:
				# import sys
				# sys.exit()
			for m, candidate in enumerate(chosen_candidates[:target_feature_indices.shape[1]]):
				target_feature_indices[i, m] = candidate

			'''
			if len(chosen_candidates) == 0:
				target_index = best.indices[i, 0]*self.hop_length
				prediction_f0.append(f0[best.indices[i, 0]])
			else:
				target_index = chosen_candidates[0]*self.hop_length
				prediction_f0.append(f0[chosen_candidates[0]])
			
			# target_feature_indices.append(target_feature_index)
			prediction += matching_audio[0, target_index:target_index + 320].tolist()
			'''
		# return target_feature_indices, prediction, prediction_f0
		return target_feature_indices
		# prev_items = prev_items[1:] + [item]


	@torch.inference_mode()
	def vocode_old(self, c: Tensor, f0: Tensor, harmonics_out_feats_weighted = None) -> Tensor:
		""" Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim) """
		
		# [1, 1, seq_len]
		# print(c.shape, f0.shape)
		f0 = f0[:c.shape[1]].unsqueeze(0).unsqueeze(0).to(self.device)
		
		# print(c.shape, f0.shape)
		# import sys
		# sys.exit()
		if harmonics_out_feats_weighted is not None:
			print("mix", c.shape, f0.shape, harmonics_out_feats_weighted.shape)
			# import sys
			# sys.exit()
			y_g_hat = self.hifigan(c, f0, harmonics_out_feats_weighted[None])
		else:
			print("wavlm_only", f0.shape)
			# import sys
			# sys.exit()
			y_g_hat = self.hifigan(c, f0)
		# print(c, y_g_hat)
		# import sys
		# sys.exit()
		# print(y_g_hat.shape)
		# import sys
		# sys.exit()
		
		y_g_hat = y_g_hat.squeeze(1)
		return y_g_hat
	

	
	# @torch.inference_mode()
	# : float | None  : float | None : str | None
	# def special_match(self, query_seq: Tensor, matching_set: Tensor, query_audio, matching_audio, shifted_query_f0, query_f0, shifted_matching_f0, matching_f0, src_wav_path, ref_wav_paths, synth_set: Tensor = None, topk: int = 4, is_going_up = True, tgt_loudness_db = -16,  target_duration = None, device = None, without_vocode = False) -> Tensor:
	def special_match(self, src_wav_file, ref_wav_file, topk: int = 4, device = None, prioritize_f0 = True, ckpt_type = "wavlm_only", tgt_loudness_db = -16, post_opt = False) -> Tensor:
		
		""" Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
		with k=`topk`. Inputs:
			- `query_seq`: Tensor (N1, dim) of the input/source query features.
			- `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
			- `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
				vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
				By default, and for best performance, this should be identical to the matching set. 
			- `topk`: k in the kNN -- the number of nearest neighbors to average over.
			- `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable. 
			- `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
			- `device`: if None, uses default device at initialization. Otherwise uses specified device
		Returns:
			- converted waveform of shape (T,)
		"""
		device = torch.device(device) if device is not None else self.device

		
		from ddsp_prematch_dataset import match_at_inference_time
		from pathlib import Path
		if "wavlm_only" not in ckpt_type and "no_harm_no_amp" not in ckpt_type:
			out_feats_weighted, harmonics_out_feats_weighted, audio_out_feats_weighted, shifted_query_f0 = match_at_inference_time(Path(src_wav_file), Path(ref_wav_file), self.wavlm, match_weights = self.weighting, synth_weights = self.weighting, topk = topk, device = device, prioritize_f0 = prioritize_f0, ckpt_type = ckpt_type, post_opt = post_opt)

			out_feats_weighted = out_feats_weighted[src_wav_file]
			harmonics_out_feats_weighted = harmonics_out_feats_weighted[src_wav_file]
			audio_out_feats_weighted = audio_out_feats_weighted[src_wav_file]
			shifted_query_f0 = shifted_query_f0[src_wav_file]
			
		
			prediction = self.vocode(out_feats_weighted[None].to(device), shifted_query_f0[None, :, None], harmonics_out_feats_weighted[None]).squeeze()
		else:
			
			out_feats_weighted, audio_out_feats_weighted, shifted_query_f0 = match_at_inference_time(Path(src_wav_file), Path(ref_wav_file), self.wavlm, match_weights = self.weighting, synth_weights = self.weighting, topk = topk, device = device, prioritize_f0 = prioritize_f0, ckpt_type = ckpt_type)
		
		
			out_feats_weighted = out_feats_weighted[src_wav_file]
			audio_out_feats_weighted = audio_out_feats_weighted[src_wav_file]
			shifted_query_f0 = shifted_query_f0[src_wav_file]

			
			# self.get_matching_set([ref_wav_file])
			# import sys
			# sys.exit()		
		
			# 
			if "wavlm_only_original" not in ckpt_type:
				prediction = self.vocode(out_feats_weighted[None].to(device), shifted_query_f0[None, :, None].to(device)).squeeze()
				# prediction = self.vocode_old(out_feats_weighted[None].to(device), shifted_query_f0).squeeze()
			else:
				prediction = self.vocode_old(out_feats_weighted[None].to(device)).squeeze()
		
		print(out_feats_weighted.dtype)
		# 
		
		# import sys
		# sys.exit()
			
		pred_wav = prediction
		
		'''
		if tgt_loudness_db is not None:
			src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
			tgt_loudness = tgt_loudness_db
			pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
		else: pred_wav = prediction
		'''

		import os
		src_identifier = os.path.basename(src_wav_file).split(".")[0]
		ref_identifier = os.path.basename(ref_wav_file).split(".")[0]
	

		# play_sequence(audio_out_feats_weighted.reshape(-1).detach().cpu().numpy())
		# play_sequence(pred_wav.detach().cpu().numpy())
		
		write_audio("/home/ken/Downloads/temp_Choral_not_used/" + src_identifier + "_to_" + ref_identifier + f"_knn_{ckpt_type}_{post_opt}.wav", pred_wav.detach().cpu().numpy(), sample_rate = 16000)
		
		# write_audio("/home/ken/Downloads/temp_Choral_not_used/raw_" + src_identifier + "_to_" + ref_identifier + "_knn_converted.wav", audio_out_feats_weighted.detach().cpu().numpy(), sample_rate = 16000)
				
				
		import sys
		sys.exit()
	
	
	
	def bulk_match(self, src_dataset_path = "/home/ken/Downloads/knn_vc_data/test", tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/test", converted_audio_dir = "/home/ken/Downloads/knn_vc_data/test_converted_audio/", topk: int = 4, device = None, prioritize_f0 = True, ckpt_type = "mix", tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/test_to_test.txt", post_opt = "no_post_opt", duration_limit = None) -> Tensor:

		# not saving pool, as there is no way to verify if it is the latest pool generating code later
		
		import os
		
		assert os.path.isdir(src_dataset_path) and os.path.isdir(tgt_dataset_path)

		from pathlib import Path
		Path(converted_audio_dir).mkdir(parents = True, exist_ok = True)
		
		src_spk_folders = sorted([item for item in Path(src_dataset_path).iterdir() if item.is_dir() and "f0_cache" not in os.path.basename(item)])
		tgt_spk_folders = sorted([item for item in Path(tgt_dataset_path).iterdir() if item.is_dir() and "f0_cache" not in os.path.basename(item)])
		
		if src_dataset_path != tgt_dataset_path:
			# to avoid error when caching pool
			assert len(set(src_spk_folders).intersection(set(tgt_spk_folders))) == 0
		
		# print(spk_folders[:10])
		# import sys
		# sys.exit()
		
		do_flattening = False
		# flatten each spk_folders
		if do_flattening:
			for spk_folder in spk_folders:
				print("Flattening:", spk_folder)
				os.system("find " + str(spk_folder) + " -mindepth 2 -type f -exec mv -t " + str(spk_folder) + " -f '{}' +")
				os.system("find " + str(spk_folder) + " -type d -empty -delete")
		# import sys
		# sys.exit()
		
		# print(len(spk_folders))
		# import sys
		# sys.exit()
		

		# for each pair, for each item in matching_pool, we match it against the entire matching_pool_1 and save the info. Now at runtime, given a file and a target_spk, we also need the target_spk's synth_pool, so save that if not already saved.
		

		with open(required_subset_file) as fp:
			import csv
			reader = csv.reader(fp, delimiter=",", quotechar='"')
			required_audio_subset = [row[2] for row_idx, row in enumerate(reader) if row_idx != 0 and row[-1] == "0"]

		
			
		# print(required_audios[0], len(required_audios))
		# import sys
		# sys.exit()


		from ddsp_prematch_dataset import match_at_inference_time
		cache_dir = "/home/ken/copies/test_cached_" + ckpt_type
		if os.path.isdir(cache_dir):
			os.system(f"rm -rf {cache_dir}")


		for i, spk_folder in enumerate(src_spk_folders):

			
			for j, tgt_spk_folder in enumerate(tgt_spk_folders):
				# avoid self to self
				if src_dataset_path == tgt_dataset_path and i == j:
					continue
					
				print(f"{spk_folder} -> {tgt_spk_folder}")
					
				
				predictions = dict()
				
				from pathlib import Path
				# print(duration_limit)
				# import sys
				# sys.exit()
				
				if "wavlm_only" not in ckpt_type and "no_harm_no_amp" not in ckpt_type:
					out_feats_weighted_collection, harmonics_out_feats_weighted_collection, audio_out_feats_weighted_collection, shifted_query_f0_collection = match_at_inference_time(Path(spk_folder), Path(tgt_spk_folder), self.wavlm, match_weights = self.weighting, synth_weights = self.weighting, topk = topk, device = device, prioritize_f0 = prioritize_f0, ckpt_type = ckpt_type, src_dataset_path = src_dataset_path, tgt_dataset_path = tgt_dataset_path, cache_dir = cache_dir, required_subset = required_audio_subset, post_opt = post_opt, duration_limit = duration_limit)
					
					print(out_feats_weighted_collection.keys())
					
					
					 
					# sorted(list(spk_folder.rglob('**/*.flac')) + list(spk_folder.rglob('**/*.wav')))
					for src_audio_file in out_feats_weighted_collection:
						src_audio_file = str(src_audio_file)
						
						out_feats_weighted = out_feats_weighted_collection[src_audio_file]
						# audio_out_feats_weighted = audio_out_feats_weighted_collection[src_audio_file]
						harmonics_out_feats_weighted = harmonics_out_feats_weighted_collection[src_audio_file]
						shifted_query_f0 = shifted_query_f0_collection[src_audio_file]
						
						prediction = self.vocode(out_feats_weighted[None].to(device), shifted_query_f0[None, :, None], harmonics_out_feats_weighted[None]).squeeze()
						
						predictions[src_audio_file] = prediction
			
						# break
				else:
					out_feats_weighted_collection, audio_out_feats_weighted_collection, shifted_query_f0_collection = match_at_inference_time(Path(spk_folder), Path(tgt_spk_folder), self.wavlm, match_weights = self.weighting, synth_weights = self.weighting, topk = topk, device = device, prioritize_f0 = prioritize_f0, ckpt_type = ckpt_type, src_dataset_path = src_dataset_path, tgt_dataset_path = tgt_dataset_path, cache_dir = cache_dir, required_subset = required_audio_subset, duration_limit = duration_limit)
				
					# sorted(list(spk_folder.rglob('**/*.flac')) + list(spk_folder.rglob('**/*.wav')))
					for src_audio_file in out_feats_weighted_collection:
						src_audio_file = str(src_audio_file)
						
						out_feats_weighted = out_feats_weighted_collection[src_audio_file]
						# audio_out_feats_weighted = audio_out_feats_weighted_collection[src_audio_file]
						shifted_query_f0 = shifted_query_f0_collection[src_audio_file]
						
						if "wavlm_only_original" in ckpt_type:
							# original does not need f0
							prediction = self.vocode(out_feats_weighted[None].to(device)).squeeze()
						else:
							prediction = self.vocode(out_feats_weighted[None].to(device), shifted_query_f0[None, :, None]).squeeze()
				
						predictions[src_audio_file] = prediction
				
				for src_audio_file in predictions:
					
					converted_audio_file = os.path.join(converted_audio_dir, os.path.basename(src_audio_file).split(".")[0], os.path.basename(tgt_spk_folder) + "." + os.path.basename(src_audio_file).split(".")[-1])
					
					Path(converted_audio_file).parent.mkdir(parents = True, exist_ok = True)
					
					# print(converted_audio_file)
					# import sys
					# sys.exit()
					
					prediction = predictions[src_audio_file]
					
					pred_wav = prediction
					# play_sequence(pred_wav.cpu().numpy(), f_s = self.sr)
					# import sys
					# sys.exit()
					
					assert len(pred_wav.shape) == 1
					# print(pred_wav.shape)
					# import sys
					# sys.exit()
					write_audio(converted_audio_file, pred_wav[None, :].cpu().numpy(), sample_rate = self.sr)

					# import sys
					# sys.exit()
