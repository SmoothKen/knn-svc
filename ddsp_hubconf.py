dependencies = ['torch', 'torchaudio', 'numpy']

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from pathlib import Path


from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.utils import AttrDict, load_checkpoint, scan_checkpoint
from ddsp_matcher import KNeighborsVC


def knn_vc(pretrained=True, progress=True, prematched=True, ckpt_type = "mix", device='cuda') -> KNeighborsVC:
	""" Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
	# using self trained ones
	# hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
	
	hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, ckpt_type, device)
	wavlm = wavlm_large(pretrained, progress, device)
	knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
	return knnvc

#  -> HiFiGAN
def hifigan_wavlm(pretrained=True, progress=True, prematched=True, ckpt_type = "mix", device='cuda'):
	""" Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
	cp = Path(__file__).parent.absolute()

	# print(str(cp/'hifigan'/'config_v1_wavlm.json'))
	# import sys
	# sys.exit()

	with open(cp/'hifigan'/'config_v1_wavlm.json') as f:
		data = f.read()
	json_config = json.loads(data)
	h = AttrDict(json_config)
	device = torch.device(device)

	if "wavlm_only" in ckpt_type or "no_harm_no_amp" in ckpt_type:
		
		if "wavlm_only_original" in ckpt_type:
			from hifigan.models import Generator as HiFiGAN
			h.hubert_dim = 1024
		else:
			from hifigan.ddsp_models_f0 import SynthesizerTrn as HiFiGAN
			h.hubert_dim = 1024
	# elif ckpt_type == "spec_only":
		# raise NotImplementedError
		# h.hubert_dim = 200
	else:
		
		# _harmonics
		from hifigan.ddsp_models import SynthesizerTrn as HiFiGAN
		h.hubert_dim = 1024
		
		
	generator = HiFiGAN(h).to(device)
	
	# always use local
	pretrained = False
	if pretrained:
		if prematched:
			url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
		else:
			url = "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt"
		state_dict_g = torch.hub.load_state_dict_from_url(
			url,
			map_location=device,
			progress=progress
		)
		generator.load_state_dict(state_dict_g['generator'])
	else:
	
		import os
		# local_ckpt_dir = "/home/ken/Downloads/knn_vc_data/ckpt"
		local_ckpt_dir = "/home/ken/Downloads/knn_vc_data/ckpt_saved"
		
		
		if os.path.isdir(local_ckpt_dir):
			# cp_g = scan_checkpoint(local_ckpt_dir, 'g_')
			
			# 'g_00850'
			cp_g = scan_checkpoint(local_ckpt_dir, ckpt_type)
			# cp_g = scan_checkpoint(local_ckpt_dir, 'g_00500000.pt')
			
			# print(cp_g)
			# import sys
			# sys.exit()
			
			
			state_dict_g = load_checkpoint(cp_g, device)
			generator.load_state_dict(state_dict_g['generator'])
			print("Loaded ckpt from local", cp_g)
		else:
			import sys
			sys.exit("Bad ckpt location")
		
	generator.eval()
	# the decoder (or the original Generator class)
	# generator.dec.remove_weight_norm()
	print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
	return generator, h


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
	"""Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
	if torch.cuda.is_available() == False:
		if str(device) != 'cpu':
			logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
			device = 'cpu'
	checkpoint = torch.hub.load_state_dict_from_url(
		"https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt", 
		map_location=device, 
		progress=progress
	)
	
	cfg = WavLMConfig(checkpoint['cfg'])
	device = torch.device(device)
	model = WavLM(cfg)
	if pretrained:
		model.load_state_dict(checkpoint['model'])
		print("Pretrained WavLM loaded")
	model = model.to(device)
	model.eval()
	print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
	return model
