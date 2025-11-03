import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .utils import init_weights, get_padding
# import pdb
# pdb.set_trace()

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
		super(ResBlock1, self).__init__()
		self.h = h
		self.convs1 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
							   padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
							   padding=get_padding(kernel_size, dilation[1]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
							   padding=get_padding(kernel_size, dilation[2])))
		])
		self.convs1.apply(init_weights)

		self.convs2 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
							   padding=get_padding(kernel_size, 1)))
		])
		self.convs2.apply(init_weights)

	def forward(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c1(xt)
			xt = F.leaky_relu(xt, LRELU_SLOPE)
			xt = c2(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for l in self.convs1:
			remove_weight_norm(l)
		for l in self.convs2:
			remove_weight_norm(l)

# Resblock1, alternate dilated conv and normal conv, 3x
# Resblock2, dilated conv, 2x
# Resblock3, dilated conv, 1x
class ResBlock2(torch.nn.Module):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
		super(ResBlock2, self).__init__()
		self.h = h
		self.convs = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
							   padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
							   padding=get_padding(kernel_size, dilation[1])))
		])
		self.convs.apply(init_weights)

	def forward(self, x):
		for c in self.convs:
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for l in self.convs:
			remove_weight_norm(l)




class ResBlock3(torch.nn.Module):
	def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
		super(ResBlock3, self).__init__()
		self.convs = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0])))
		])
		self.convs.apply(init_weights)

	def forward(self, x):
		for c in self.convs:
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for l in self.convs:
			remove_weight_norm(l)


def upsample(signal, factor):
	signal = signal.permute(0, 2, 1)
	signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
	return signal.permute(0, 2, 1)



class Generator(torch.nn.Module):
	def __init__(self, hps):
		super(Generator, self).__init__()
		
		self.hps = hps
		self.lin_pre = nn.Linear(hps.hubert_dim, hps.hifi_dim)
		self.num_kernels = len(hps.resblock_kernel_sizes)
		self.num_upsamples = len(hps.upsample_rates)
		
		self.conv_pre = Conv1d(hps.hifi_dim, hps.upsample_initial_channel, 7, 1, padding=3)


		resblock = ResBlock1 if hps.resblock == '1' else ResBlock2


		self.downs = nn.ModuleList()
		for i, (u, k) in enumerate(zip(hps.upsample_rates, hps.upsample_kernel_sizes)):
			# downsampling for ddsp waveform, hence everything reversed
			# i = len(hps.upsample_rates) - 1 - i
			j = len(hps.upsample_rates) - 1 - i
			# u = hps.upsample_rates[i]
			# k = hps.upsample_kernel_sizes[i]
			u = hps.upsample_rates[j]
			k = hps.upsample_kernel_sizes[j]
			
			# self.downs.append(weight_norm(Conv1d(hps.n_harmonic + 2, hps.n_harmonic + 2, k, u, padding=k//2)))
			# print(hps.n_harmonic*(2**i))
			
			self.downs.append(weight_norm(Conv1d(hps.n_harmonic*(2**i), hps.n_harmonic*(2**(i+1)), k, u, padding=k//2)))
		
		self.resblocks_downs = nn.ModuleList()
		for i in range(len(self.downs)):
			# j = len(hps.upsample_rates) - 1 - i
			# self.resblocks_downs.append(ResBlock3(hps.n_harmonic + 2, 3, (1, 3)))
			
			self.resblocks_downs.append(ResBlock3(hps.n_harmonic*(2**(i+1)), 3, (1, 3)))
			
			
		# self.concat_pre = Conv1d(hps.upsample_initial_channel + hps.n_harmonic + 2, hps.upsample_initial_channel, 3, 1, padding=1)
		self.concat_pre = Conv1d(hps.upsample_initial_channel*2, hps.upsample_initial_channel, 3, 1, padding=1)
		self.concat_conv = nn.ModuleList()
		for i in range(len(hps.upsample_rates)):
			ch = hps.upsample_initial_channel//(2**(i+1))
			# self.concat_conv.append(Conv1d(ch + hps.n_harmonic + 2, ch, 3, 1, padding=1, bias=False))
			self.concat_conv.append(Conv1d(ch*2, ch, 3, 1, padding=1, bias=False))


		self.ups = nn.ModuleList()
		for i, (u, k) in enumerate(zip(hps.upsample_rates, hps.upsample_kernel_sizes)):
			self.ups.append(weight_norm(
				ConvTranspose1d(hps.upsample_initial_channel//(2**i), hps.upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2)))


		self.resblocks = nn.ModuleList()
		for i in range(len(self.ups)):
			ch = hps.upsample_initial_channel//(2**(i+1))
			for j, (k, d) in enumerate(zip(hps.resblock_kernel_sizes, hps.resblock_dilation_sizes)):
				self.resblocks.append(resblock(hps, ch, k, d))


		self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
		# original HiFi-GAN
		# self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
		self.ups.apply(init_weights)
		self.conv_post.apply(init_weights)



	def forward(self, x, ddsp):
		x = self.lin_pre(x)
		x = x.permute(0, 2, 1) # (bs, seq_len, dim) --> (bs, dim, seq_len)


		x = self.conv_pre(x)

		se = ddsp
		res_features = [se]
		# print("!!!", se.shape)
		
		for i in range(self.num_upsamples):
			in_size = se.size(2)
			se = self.downs[i](se)
			
			# print(se.shape)
			se = self.resblocks_downs[i](se)
			up_rate = self.hps.upsample_rates[self.num_upsamples - 1 - i]
			se = se[:, :, : in_size // up_rate]
			res_features.append(se)
			# print("---", se.shape)
		# torch.Size([16, 512, 22]) torch.Size([16, 34, 22])
		# print(x.shape, se.shape, ddsp.shape)
		# import sys
		# sys.exit()
		
		# print(x.shape, se.shape)
		x = torch.cat([x, se], 1)
		x = self.concat_pre(x)

		for i in range(self.num_upsamples):
			x = F.leaky_relu(x, LRELU_SLOPE)
			in_size = x.size(2)
			x = self.ups[i](x)
			
			# print(x.shape, res_features[self.num_upsamples - 1 - i].shape)
			# import sys
			# sys.exit()
			# doing an extra step to ensure the upsampling ratio is exact, but may not be necessary assuming ddsp downsampling division is exact
			# x = x[:, :, : in_size * self.upsample_rates[i]]
			

			x = torch.cat([x, res_features[self.num_upsamples - 1 - i]], 1)
			x = self.concat_conv[i](x)

			xs = None
			for j in range(self.num_kernels):
				if xs is None:
					xs = self.resblocks[i*self.num_kernels+j](x)
				else:
					xs += self.resblocks[i*self.num_kernels+j](x)
			x = xs / self.num_kernels
				
		x = F.leaky_relu(x)
		x = self.conv_post(x)
		x = torch.tanh(x)
		
		return x


	def remove_weight_norm(self):
		print('Removing weight norm...')
		for l in self.ups:
			remove_weight_norm(l)
		for l in self.resblocks:
			l.remove_weight_norm()



class ConvReluNorm(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
		super().__init__()
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
		assert n_layers > 1, "Number of layers should be larger than 0."

		self.conv_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
		self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
		self.norm_layers.append(LayerNorm(hidden_channels))
		self.relu_drop = nn.Sequential(
				nn.ReLU(),
				nn.Dropout(p_dropout))
		for _ in range(n_layers-1):
			self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
			self.norm_layers.append(LayerNorm(hidden_channels))
		self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
		self.proj.weight.data.zero_()
		self.proj.bias.data.zero_()

	def forward(self, x):
		x = self.conv_layers[0](x)
		x = self.norm_layers[0](x)
		x = self.relu_drop(x)

		for i in range(1, self.n_layers):
			x_ = self.conv_layers[i](x)
			x_ = self.norm_layers[i](x_)
			x_ = self.relu_drop(x_)
			x = (x + x_) / 2
		x = self.proj(x)
		return x


		
class Generator_Harm(torch.nn.Module):
	def __init__(self, hps):
		super(Generator_Harm, self).__init__()
		self.hps = hps

		self.prenet = Conv1d(hps.model.hidden_channels, hps.model.hidden_channels, 3, padding=1)
				
		self.net = ConvReluNorm(hps.model.hidden_channels,
						hps.model.hidden_channels,
						hps.model.hidden_channels,
						hps.model.kernel_size,
						8,
						hps.model.p_dropout)

		self.postnet = Conv1d(hps.model.hidden_channels, hps.model.n_harmonic+1, 3, padding=1)

	def forward(self, f0, harm):
		pitch = f0.transpose(1, 2)
		harm = self.prenet(harm)

		harm = self.net(harm)

		harm = self.postnet(harm)
		harm = harm.transpose(1, 2)
		param = harm

		param = scale_function(param)
		total_amp = param[..., :1]
		amplitudes = param[..., 1:]
		amplitudes = remove_above_nyquist(
				amplitudes,
				pitch,
				self.hps.data.sample_rate,
		)
		amplitudes /= amplitudes.sum(-1, keepdim=True)
		amplitudes *= total_amp

		amplitudes = upsample(amplitudes, self.hps.data.hop_size)
		pitch = upsample(pitch, self.hps.data.hop_size)

		n_harmonic = amplitudes.shape[-1]
		omega = torch.cumsum(pitch.double() / self.hps.data.sample_rate, dim=1)
		# round to [-0.5, 0.5] (to avoid cumsum getting too large)
		omega = (2 * math.pi * (omega - torch.round(omega))).float()
		
		omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
		signal_harmonics = (torch.sin(omegas) * amplitudes)
		signal_harmonics = signal_harmonics.transpose(1, 2)
		return signal_harmonics
		
		
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

		
		
def play_sequence(audio_chunk, f_s = 16000):
	import sounddevice as sd
	sd.play(audio_chunk, f_s, blocking = True)


import sys, os
from pathlib import Path
sys.path.append(Path(os.path.abspath(__file__)))
from ddsp_prematch_dataset import get_bulk_dsp_choral

# from .ddsp_meldataset import get_bulk_dsp_choral
class SynthesizerTrn(nn.Module):
	def __init__(self, hps):
		super().__init__()
		self.hps = hps

		self.dec = Generator(hps)
		
		# self.dec_harm = Generator_Harm(hps)
		# self.dec_noise = Generator_Noise(hps)  
		
		# self.sin_prenet = nn.Conv1d(1, hps.n_harmonic + 2, 3, padding=1)
		self.sin_prenet = nn.Conv1d(1, hps.n_harmonic, 3, padding=1)
		

	def forward(self, mel, f0, harmonics_out_feats_weighted):


		# print(f0.shape, harmonics_out_feats_weighted.shape)
		# import sys
		# sys.exit()
		
		
		# dsp_signal = get_bulk_dsp_choral(f0[None, :, None], harmonics_out_feats_weighted, sample_rate = self.sampling_rate, hop_size = self.hop_size, dsp_type = "sin").reshape(-1)
		# print(f0.shape, harmonics_out_feats_weighted.shape)
		# import sys
		# sys.exit()
		# output (batch, time, 1) -> tranpose -> (batch, 1, time)
		f0_wave = get_bulk_dsp_choral(f0, harmonics_out_feats_weighted, sample_rate = self.hps.sampling_rate, hop_size = self.hps.hop_size).transpose(1, 2)
		

		
		'''
		if f0_wave.shape[-1] > 8000:
			print(f0_wave.shape)
			write_audio("/tmp/super_temp.mp3", f0_wave[0], 16000)
			import sys
			sys.exit()
		'''
		
		
		'''
		
		# print(dsp_signal.shape)
		# import sys
		# sys.exit()
		
		# pitch = upsample(f0.transpose(1, 2), self.hps.data.hop_size)
		# omega = torch.cumsum(2 * math.pi * pitch / self.hps.data.sample_rate, 1)
		
		# assume mel and f0 have correspondence, and final waveform length has the same length as f0
		
		pitch = upsample(f0.transpose(1, 2), self.hps.hop_size)
		
		# print(f0.shape, pitch.shape)
		# import sys
		# sys.exit()
		omega = torch.cumsum(pitch.double() / self.hps.sampling_rate, dim=1)
		import math
		omega = (2 * math.pi * (omega - torch.round(omega))).float()
		f0_wave = torch.sin(omega).transpose(1, 2)
		'''
		
		# dsp synthesize
		# noise_x = self.dec_noise(p_z, y_mask)
		# harm_x = self.dec_harm(F0, p_z, y_mask)

		# dsp waveform
		# dsp_o = torch.cat([harm_x, noise_x], axis=1)

		# decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1) 
		# (batch_size, channel, frames)   
		decoder_condition = self.sin_prenet(f0_wave)

		
		
		# dsp based HiFiGAN vocoder
		# x_slice, ids_slice = commons.rand_slice_segments(p_z, bn_lengths, self.hps.train.segment_size // self.hps.data.hop_size)
		# F0_slice = commons.slice_segments(F0, ids_slice, self.hps.train.segment_size // self.hps.data.hop_size)
		# dsp_slice = commons.slice_segments(dsp_o, ids_slice * self.hps.data.hop_size, self.hps.train.segment_size)
		# condition_slice = commons.slice_segments(decoder_condition, ids_slice * self.hps.data.hop_size, self.hps.train.segment_size)
		# print(mel.shape)
		generated_waveform = self.dec(mel, decoder_condition)
		
		# print(f0_wave.shape, decoder_condition.shape, generated_waveform.shape)
		# import sys
		# sys.exit()
		
		# , f0_wave
		return generated_waveform


class DiscriminatorP(torch.nn.Module):
	def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
		super(DiscriminatorP, self).__init__()
		self.period = period
		norm_f = weight_norm if use_spectral_norm == False else spectral_norm
		self.convs = nn.ModuleList([
			norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
			norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
		])
		self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

	def forward(self, x):
		fmap = []

		# 1d to 2d
		b, c, t = x.shape
		if t % self.period != 0: # pad first
			n_pad = self.period - (t % self.period)
			x = F.pad(x, (0, n_pad), "reflect")
			t = t + n_pad
		x = x.view(b, c, t // self.period, self.period)

		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, LRELU_SLOPE)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)

		return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
	def __init__(self):
		super(MultiPeriodDiscriminator, self).__init__()
		self.discriminators = nn.ModuleList([
			DiscriminatorP(2),
			DiscriminatorP(3),
			DiscriminatorP(5),
			DiscriminatorP(7),
			DiscriminatorP(11),
		])

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
	def __init__(self, use_spectral_norm=False):
		super(DiscriminatorS, self).__init__()
		norm_f = weight_norm if use_spectral_norm == False else spectral_norm
		self.convs = nn.ModuleList([
			norm_f(Conv1d(1, 128, 15, 1, padding=7)),
			norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
			norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
			norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
			norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
			norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
			norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
		])
		self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

	def forward(self, x):
		fmap = []
		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, LRELU_SLOPE)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)

		return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
	def __init__(self):
		super(MultiScaleDiscriminator, self).__init__()
		self.discriminators = nn.ModuleList([
			DiscriminatorS(use_spectral_norm=True),
			DiscriminatorS(),
			DiscriminatorS(),
		])
		self.meanpools = nn.ModuleList([
			AvgPool1d(4, 2, padding=2),
			AvgPool1d(4, 2, padding=2)
		])

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			if i != 0:
				y = self.meanpools[i-1](y)
				y_hat = self.meanpools[i-1](y_hat)
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
	loss = 0
	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):
			loss += torch.mean(torch.abs(rl - gl))

	return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []
	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		r_loss = torch.mean((1-dr)**2)
		g_loss = torch.mean(dg**2)
		loss += (r_loss + g_loss)
		r_losses.append(r_loss.item())
		g_losses.append(g_loss.item())

	return loss, r_losses, g_losses


def generator_loss(disc_outputs):
	loss = 0
	gen_losses = []
	for dg in disc_outputs:
		l = torch.mean((1-dg)**2)
		gen_losses.append(l)
		loss += l

	return loss, gen_losses

