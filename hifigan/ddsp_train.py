import argparse
import itertools
import json
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# from .meldataset import (LogMelSpectrogram, MelDataset, get_dataset_filelist, mel_spectrogram)
from .ddsp_meldataset import (LogMelSpectrogram, MelDataset, get_dataset_filelist, mel_spectrogram)
						 
# Generator, Generator_Harm, 
from .utils import (AttrDict, build_env, load_checkpoint, plot_spectrogram,
					save_checkpoint, scan_checkpoint)


torch.backends.cudnn.benchmark = True
USE_ALT_MELCALC = True


def train(rank, a, h):
	if h.num_gpus > 1:
		init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
						   world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

	torch.cuda.manual_seed(h.seed)
	device = torch.device('cuda:{:d}'.format(rank))

	train_df, valid_df = get_dataset_filelist(a)

	from pathlib import Path
	# "train-clean-100"
	trainset = MelDataset(h, h.segment_size, h.n_fft, h.num_mels,
						  h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
						  shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
						  fine_tuning=a.fine_tuning,
						  # audio_root_path=Path(a.audio_root_path)/"train", feat_root_path=Path(a.feature_root_path)/"train", 
						  audio_root_path=Path(a.audio_root_path_train),
						  feat_root_path=Path(a.feature_root_path_train),
						  use_alt_melcalc=USE_ALT_MELCALC)

	train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

	train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
							  sampler=train_sampler,
							  batch_size=h.batch_size,
							  pin_memory=True,
							  drop_last=True)

	alt_melspec = LogMelSpectrogram(h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax).to(device)

	# print(h.num_workers)
	# import sys
	# sys.exit()

	if rank == 0:
		# "dev-clean"
		validset = MelDataset(h, h.segment_size, h.n_fft, h.num_mels,
							  h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
							  fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
							  # audio_root_path=Path(a.audio_root_path)/"valid", feat_root_path=Path(a.feature_root_path)/"valid", 
							  audio_root_path=Path(a.audio_root_path_valid),
							  feat_root_path=Path(a.feature_root_path_valid),
							  
							  use_alt_melcalc=USE_ALT_MELCALC)
		validation_loader = DataLoader(validset, num_workers=4, shuffle=False,
									   sampler=None,
									   batch_size=1,
									   pin_memory=True,
									   drop_last=True)

		sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))


	if h.with_harm:
		from .ddsp_models import (SynthesizerTrn,
						 MultiPeriodDiscriminator,
						 MultiScaleDiscriminator, discriminator_loss, feature_loss,
						 generator_loss)
	else:
		from .ddsp_models_f0 import (SynthesizerTrn,
						 MultiPeriodDiscriminator,
						 MultiScaleDiscriminator, discriminator_loss, feature_loss,
						 generator_loss)

	
	assert type(h.with_amp_ratio) == bool and type(h.with_harm) == bool, [type(h.with_amp_ratio), type(h.with_harm)]
	
	print(["with_amp_ratio", h.with_amp_ratio, "with_harm", h.with_harm])
	import time
	time.sleep(5)
	# import sys
	# sys.exit()
	
	# generator = Generator(h).to(device)
	generator = SynthesizerTrn(h).to(device)
	mpd = MultiPeriodDiscriminator().to(device)
	msd = MultiScaleDiscriminator().to(device)

	if rank == 0:
		print(generator)
		os.makedirs(a.checkpoint_path, exist_ok=True)
		print("checkpoints directory : ", a.checkpoint_path)

	if os.path.isdir(a.checkpoint_path):
		cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
		cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
		
		# comment this out if do not want to retrain by default
		cp_g = None
		cp_do = None
		
	steps = 0
	if cp_g is None or cp_do is None:
		state_dict_do = None
		last_epoch = -1
	else:
		state_dict_g = load_checkpoint(cp_g, device)
		state_dict_do = load_checkpoint(cp_do, device)
		generator.load_state_dict(state_dict_g['generator'])
		mpd.load_state_dict(state_dict_do['mpd'])
		msd.load_state_dict(state_dict_do['msd'])
		steps = state_dict_do['steps'] + 1
		last_epoch = state_dict_do['epoch']
		print(f"Restored checkpoint from {cp_g} and {cp_do}")

	if h.num_gpus > 1:
		print("Multi-gpu detected")
		generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
		mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
		msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

	optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
	optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
								h.learning_rate, betas=[h.adam_b1, h.adam_b2])

	if state_dict_do is not None:
		optim_g.load_state_dict(state_dict_do['optim_g'])
		optim_d.load_state_dict(state_dict_do['optim_d'])

	scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
	scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
	
	
	if a.fp16:
		scaler_g = GradScaler()
		scaler_d = GradScaler()
		
		
	generator.train()
	mpd.train()
	msd.train()
	prev_min_val_err = 9999
	prev_min_val_err_step = -1
	cur_best_ckpt_paths = []
	
	
	if rank == 0: mb = master_bar(range(max(0, last_epoch), a.training_epochs))
	else: mb = range(max(0, last_epoch), a.training_epochs)

	for epoch in mb:
		
		# max_epoch
		if steps > 1000000:
			break
		
		if rank == 0:
			start = time.time()
			mb.write("Epoch: {} ".format(epoch+1) + "Steps: {}".format(steps))

		if h.num_gpus > 1:
			train_sampler.set_epoch(epoch)

		if rank == 0: pb = progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb)
		else: pb = enumerate(train_loader)
		

		for i, batch in pb:
			# print("Hello", batch[0].shape)
			if rank == 0:
				start_b = time.time()
			x, y, _, y_mel, f0, harmonics_out_feats_weighted = batch
			x = x.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			y_mel = y_mel.to(device, non_blocking=True)
			f0 = f0.to(device, non_blocking=True)
			harmonics_out_feats_weighted = harmonics_out_feats_weighted.to(device, non_blocking=True)
			y = y.unsqueeze(1)
			
			with torch.cuda.amp.autocast(enabled=a.fp16):
				if h.with_harm:
					y_g_hat = generator(x, f0, harmonics_out_feats_weighted)
				else:
					y_g_hat = generator(x, f0)
					
					
				if USE_ALT_MELCALC:
					y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
				else:
					y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
											h.fmin, h.fmax_for_loss)
			# print(x.shape, y_g_hat.shape, y_g_hat_mel.shape, y_mel.shape, y.shape)
			optim_d.zero_grad()
			# print(y.shape, y_g_hat.shape)
			# import sys
			# sys.exit()
			
			with torch.cuda.amp.autocast(enabled=a.fp16):
				# MPD
				y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
				loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

				# MSD
				y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
				loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

				loss_disc_all = loss_disc_s + loss_disc_f

			if a.fp16: 
				scaler_d.scale(loss_disc_all).backward()
				scaler_d.step(optim_d)
				scaler_d.update()
			else: 
				loss_disc_all.backward()
				optim_d.step()

			# Generator
			optim_g.zero_grad()

			with torch.cuda.amp.autocast(enabled=a.fp16):
				# L1 Mel-Spectrogram Loss
				loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

				y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
				y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
				loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
				loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
				loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
				loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
				loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

			if a.fp16:
				scaler_g.scale(loss_gen_all).backward()
				scaler_g.step(optim_g)
				scaler_g.update()
			else:
				loss_gen_all.backward()
				optim_g.step()

			if rank == 0:
				
				# print(steps)
				
				# STDOUT logging
				if steps % a.stdout_interval == 0:
					with torch.no_grad():
						mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

					'''
					mb.write('Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB'. \
							format(steps, loss_gen_all, mel_error, time.time() - start_b, torch.cuda.max_memory_allocated()/1e9))
					'''
					
					mb.write('Steps:{:,d}, Gen Loss:{:4.3f}, Mel-Spec. Error:{:4.3f}, prev_min_val_err:{:4.3f} at {:,d}'. \
							format(steps, loss_gen_all, mel_error, prev_min_val_err, prev_min_val_err_step) + f" with_harm: {h.with_harm}, with_amp_ratio: {h.with_amp_ratio}, dataset: {os.path.basename(a.audio_root_path_train)}")
					
					# mb.child.comment = "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}". \
							# format(steps, loss_gen_all, mel_error)
					
			
			
				# Tensorboard summary logging
				if steps % a.summary_interval == 0:
					sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
					sw.add_scalar("training/mel_spec_error", mel_error, steps)
					sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)

				# print("Hellllo")
				# Validation
				if steps % a.validation_interval == 0:  # and steps != 0:
					generator.eval()
					torch.cuda.empty_cache()
					val_err_tot = 0
					with torch.no_grad():
						for j, batch in progress_bar(enumerate(validation_loader), total=len(validation_loader), parent=mb):
							x, y, _, y_mel, f0, harmonics_out_feats_weighted = batch
							
							if h.with_harm:
								y_g_hat = generator(x.to(device), f0.to(device), harmonics_out_feats_weighted.to(device))
							else:
								y_g_hat = generator(x.to(device), f0.to(device))
								
							y_mel = y_mel.to(device, non_blocking=True)
							
							# print(x.shape, y.shape, y_mel.shape, y_g_hat.shape)
							
							if USE_ALT_MELCALC:
								y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
								if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
									# pad it 
									n_pad = h.hop_size 
									y_g_hat = F.pad(y_g_hat, (n_pad//2, n_pad - n_pad//2))
									y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
							else:
								y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
															h.hop_size, h.win_size,
															h.fmin, h.fmax_for_loss)
							#print('valid', x.shape, y_g_hat.shape, y_g_hat_mel.shape, y_mel.shape, y.shape)
							val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

							if j <= 4:
								if steps == 0:
									sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
									sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

								sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
								if USE_ALT_MELCALC:
									y_hat_spec = alt_melspec(y_g_hat.squeeze(1))
								else:
									y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
																h.hop_size, h.win_size,
																h.fmin, h.fmax_for_loss)

								sw.add_figure('generated/y_hat_spec_{}'.format(j),
											  plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

						val_err = val_err_tot / (j+1)
						sw.add_scalar("validation/mel_spec_error", val_err, steps)
						mb.write(f"validation run complete at {steps:,d} steps. validation mel spec error: {val_err:5.4f}")
						# if prev_min_val_err is None or prev_min_val_err > val_err:
							# prev_min_val_err = val_err
							# prev_min_val_err_step = steps



					if prev_min_val_err is None or prev_min_val_err > val_err:
						prev_min_val_err = val_err
						prev_min_val_err_step = steps
						
						
						# checkpointing
						# if steps % a.checkpoint_interval == 0 and steps != 0:
						temp_best_ckpt_paths = []
						checkpoint_path = "{}/g_{:08d}.pt".format(a.checkpoint_path, steps)
						save_checkpoint(checkpoint_path,
										{'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
						
						
						temp_best_ckpt_paths.append(checkpoint_path)
						
						checkpoint_path = "{}/do_{:08d}.pt".format(a.checkpoint_path, steps)
						save_checkpoint(checkpoint_path, 
										{'mpd': (mpd.module if h.num_gpus > 1
															 else mpd).state_dict(),
										 'msd': (msd.module if h.num_gpus > 1
															 else msd).state_dict(),
										 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
										 'epoch': epoch})
						temp_best_ckpt_paths.append(checkpoint_path)
						
						for path in cur_best_ckpt_paths:
							os.system(f"rm {path}")
						
						cur_best_ckpt_paths = temp_best_ckpt_paths
									
				



					# print("Hellllo_2")
					generator.train()
					sw.add_scalar("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9, steps)
					sw.add_scalar("memory/max_reserved_gb", torch.cuda.max_memory_reserved()/1e9, steps)
					torch.cuda.reset_peak_memory_stats()
					torch.cuda.reset_accumulated_memory_stats()

			steps += 1

		scheduler_g.step()
		scheduler_d.step()
		
		if rank == 0:
			print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
	print('Initializing Training Process..')

	parser = argparse.ArgumentParser()

	parser.add_argument('--group_name', default=None)
	parser.add_argument('--audio_root_path_train', required=True)
	parser.add_argument('--audio_root_path_valid', required=True)
	parser.add_argument('--feature_root_path_train', required=True)
	parser.add_argument('--feature_root_path_valid', required=True)
	parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
	parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
	parser.add_argument('--checkpoint_path', default='cp_hifigan')
	parser.add_argument('--config', default='')
	parser.add_argument('--training_epochs', default=15000, type=int)
	parser.add_argument('--stdout_interval', default=5, type=int)
	parser.add_argument('--checkpoint_interval', default=5000, type=int)
	parser.add_argument('--summary_interval', default=25, type=int)
	parser.add_argument('--validation_interval', default=1000, type=int)
	parser.add_argument('--fp16', default=False, type=bool)
	parser.add_argument('--fine_tuning', action='store_true')

	a = parser.parse_args()
	print(a)
	with open(a.config) as f:
		data = f.read()

	json_config = json.loads(data)
	h = AttrDict(json_config)
	build_env(a.config, 'config.json', a.checkpoint_path)

	torch.manual_seed(h.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(h.seed)
		h.num_gpus = torch.cuda.device_count()
		h.batch_size = int(h.batch_size / h.num_gpus)
		print('Batch size per GPU :', h.batch_size)
	else:
		pass
		
	h.num_gpus = torch.cuda.device_count() - 1
	print("Number of GPUs:", h.num_gpus)

	if h.num_gpus > 1:
		mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
	else:
		train(0, a, h)


if __name__ == '__main__':
	main()



# python3 -m hifigan.ddsp_train --audio_root_path_train /home/ken/Downloads/knn_vc_data/train --audio_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria/valid --feature_root_path_train /home/ken/Downloads/knn_vc_data/cached/train --feature_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria_cached/valid --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path /home/ken/Downloads/knn_vc_data/ckpt --fp16 False --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 1800 --fine_tuning

# python3 -m hifigan.ddsp_train --audio_root_path_train /home/ken/Downloads/knn_vc_data/OpenSinger_train --audio_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria/valid --feature_root_path_train /home/ken/Downloads/knn_vc_data/cached/OpenSinger_train --feature_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria_cached/valid --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path /home/ken/Downloads/knn_vc_data/ckpt --fp16 False --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 1800 --fine_tuning
