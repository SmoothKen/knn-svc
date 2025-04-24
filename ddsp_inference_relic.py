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


import torch, torchaudio

# knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
import sys
if len(sys.argv) < 3:
	print("Enter content wav as \$1, and style wavs as \$2...")
	
# "spec_only",
assert any(item in sys.argv[-1] for item in {"wavlm_only", "mix"}), "Bad sys.argv[-1]"
feature_type = sys.argv[-1]



from ddsp_prematch_dataset import match_at_inference_time
src_wav_path = sys.argv[1]
ref_wav_paths = sys.argv[2:-1]


from ddsp_hubconf import knn_vc
knn_vc = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda', feature_type = feature_type)


# WavLM ckpt: /home/ken/.cache/torch/hub/checkpoints/WavLM-Large.pt
# HiFi-GAN ckpt: /home/ken/.cache/torch/hub/checkpoints/prematch_g_02500000.pt

# Or, if you would like the vocoder trained not using prematched data, set prematched=False.



if "wavlm_only" not in feature_type and "no_harm_no_amp" not in feature_type:
	assert sys.argv[-2].startswith("post_opt") or  sys.argv[-2].startswith("no_post_opt")
	post_opt = sys.argv[-2]
	# post_opt = (sys.argv[-2] == "post_opt")	
else:
	post_opt = "no_post_opt"


# single file trials
src_wav_path = sys.argv[1]
ref_wav_path = sys.argv[2]
out_wav = knn_vc.special_match(src_wav_file = src_wav_path, ref_wav_file = ref_wav_path, topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, post_opt = post_opt)

import sys
sys.exit()



# python ./ddsp_inference.py test_special post_opt_0.2 mix_harm_no_amp_0.634; python ./ddsp_inference.py xx post_opt_0.2 mix_harm_no_amp_0.552


# python ./ddsp_inference.py xx 5 post_opt_0.2 mix_harm_no_amp_0.552; python ./ddsp_inference.py xx 10 post_opt_0.2 mix_harm_no_amp_0.552; python ./ddsp_inference.py xx 30 post_opt_0.2 mix_harm_no_amp_0.552; python ./ddsp_inference.py xx 60 post_opt_0.2 mix_harm_no_amp_0.552; python ./ddsp_inference.py xx 90 post_opt_0.2 mix_harm_no_amp_0.552


# for multiple files conversion, just put each in folder of folder (dataset).
import sys
if sys.argv[1] == "test_special":
	if sys.argv[2] in ["5", "10", "30", "60", "90"]:
		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/test", converted_audio_dir = f"/home/ken/Downloads/knn_vc_data/duration_limit_{int(sys.argv[2])}_test_converted_audio_" + feature_type + f"_{post_opt}/", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/test_to_test.txt", post_opt = post_opt, duration_limit = int(sys.argv[2]))
	else:
		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/test", converted_audio_dir = "/home/ken/Downloads/knn_vc_data/test_converted_audio_" + feature_type + f"_{post_opt}/", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/test_to_test.txt", post_opt = post_opt, duration_limit = None)
		
else:
	# print(sys.argv[2])
	# import sys
	# sys.exit()
	
	if sys.argv[2] in ["5", "10", "30", "60", "90"]:
		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test", converted_audio_dir = f"/home/ken/Downloads/knn_vc_data/duration_limit_{int(sys.argv[2])}_OpenSinger_test_converted_audio_" + feature_type + f"_{post_opt}", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt", post_opt = post_opt, duration_limit = int(sys.argv[2]))

		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/nus-smc-corpus_48", converted_audio_dir = f"/home/ken/Downloads/knn_vc_data/duration_limit_{int(sys.argv[2])}_OpenSinger_test_to_nus-smc-corpus_48_audio_" + feature_type + f"_{post_opt}/", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt", post_opt = post_opt, duration_limit = int(sys.argv[2]))

		
	else:
		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test", converted_audio_dir = "/home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_" + feature_type + f"_{post_opt}/", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt", post_opt = post_opt, duration_limit = None)

		knn_vc.bulk_match(src_dataset_path = "/home/ken/Downloads/knn_vc_data/OpenSinger_test",  tgt_dataset_path = "/home/ken/Downloads/knn_vc_data/nus-smc-corpus_48", converted_audio_dir = "/home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_" + feature_type + f"_{post_opt}/", topk = 4, device = "cuda", prioritize_f0 = True, feature_type = feature_type, tgt_loudness_db = -16, required_subset_file = "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt", post_opt = post_opt, duration_limit = None)


import sys
sys.exit()



import librosa, os
if librosa.get_samplerate(src_wav_path) != 16000:
	if not os.path.isfile(src_wav_path[:-4] + "_resampled_16000.wav"):
		assert src_wav_path[-4:] == ".wav"
		os.system("ffmpeg -i " + src_wav_path + " -osr 16000 " + src_wav_path[:-4] + "_resampled_16000.wav")
	
	src_wav_path = src_wav_path[:-4] + "_resampled_16000.wav"
	
temp_ref_wav_paths = []
for ref_wav_path in ref_wav_paths:
	if librosa.get_samplerate(ref_wav_path) != 16000:
		if not os.path.isfile(ref_wav_path[:-4] + "_resampled_16000.wav"):
			assert ref_wav_path[-4:] == ".wav"
			os.system("ffmpeg -i " + ref_wav_path + " -osr 16000 " + ref_wav_path[:-4] + "_resampled_16000.wav")

		temp_ref_wav_paths.append(ref_wav_path[:-4] + "_resampled_16000.wav")
	else:
		temp_ref_wav_paths.append(ref_wav_path)
		
ref_wav_paths = temp_ref_wav_paths
print("src:", src_wav_path)
print("ref:", ref_wav_paths)

# sys.exit()


query_seq, query_audio = knn_vc.get_features(src_wav_path, return_audio = True)
matching_set = knn_vc.get_matching_set(ref_wav_paths)

# also, try both switch to vad_level = 7
_, matching_audio = knn_vc.get_features(ref_wav_paths[0], return_audio = True)
print("Should match", matching_audio.shape[1]//320)
# matching_set = query_seq[len(query_seq)//2:]
# query_seq = query_seq[:len(query_seq)//2]
matching_set = matching_set.to("cuda")






'''
print(torch.max(torch.abs(torch.mean(query_seq[len(query_seq)//2:], dim = 0) - torch.mean(query_seq[:len(query_seq)//2], dim = 0))))
print(torch.max(torch.abs(torch.mean(query_seq[len(query_seq)//2:], dim = 0) - torch.mean(matching_set[:len(query_seq)//2], dim = 0))))
import sys
sys.exit()
'''


# query_seq = query_seq - torch.mean(query_seq, dim = 0) + torch.mean(matching_set, dim = 0)


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
	
'''	
import numpy as np
plot_matrix(query_seq.cpu().numpy().T, row_names = None, col_names = 320*np.arange(query_seq.shape[0])/16000)
import sys
sys.exit()
'''




# torch.Size([1941, 1024])
# query_seq_diff = torch.diff(query_seq, dim = 0)
# matching_set_diff = torch.diff(matching_set, dim = 0)

# print(query_seq_diff.shape, matching_set.shape)
# import sys
# sys.exit()
# matching_set = 

# print(query_seq.shape, matching_set.shape)
# import sys
# sys.exit()

query_f0 = knn_vc.get_f0(src_wav_path)
matching_f0 = knn_vc.get_multiple_f0(ref_wav_paths)


query_f0_median = torch.median(torch.log(query_f0[query_f0 != 0]))
matching_f0_median = torch.median(torch.log(matching_f0[matching_f0 != 0]))


print("query f0 median", torch.exp(query_f0_median), "shape", query_f0.shape)
print("ref f0 median", torch.exp(matching_f0_median), "shape", matching_f0.shape)



import copy

# query_f0_old = copy.deepcopy(query_f0)
shifted_query_f0 = copy.deepcopy(query_f0)
shifted_query_f0[query_f0 != 0] = torch.exp(torch.log(query_f0[query_f0 != 0]) + matching_f0_median - query_f0_median)

shifted_matching_f0 = copy.deepcopy(matching_f0)
shifted_matching_f0[matching_f0 != 0] = torch.exp(torch.log(matching_f0[matching_f0 != 0]) - matching_f0_median + query_f0_median)


# query_f0 = query_f0[:len(query_f0)//2]


# generic f0 based female?


print(query_seq.shape, matching_set.shape)
# sys.exit()
'''
fake_diff = knn_vc.match(query_seq_diff, matching_set_diff, query_f0, topk=1, without_vocode = True)
fake_out_feats = torch.cumsum(torch.cat([query_seq[:1], fake_diff], dim = 0), dim = 0)
assert query_seq.shape == fake_out_feats.shape

out_wav = knn_vc.vocode(fake_out_feats[None].to("cuda"), query_f0).cpu().squeeze()
'''

# out_wav = knn_vc.match(query_seq, matching_set, query_f0, topk=4)
out_wav = knn_vc.match(query_seq, matching_set, shifted_query_f0, topk=4)

# out_wav = knn_vc.self_match(query_seq, matching_set, query_f0, topk=4)
# out_wav = knn_vc.special_match(query_seq, matching_set, src_wav_path = src_wav_path, ref_wav_paths = ref_wav_paths, query_audio = query_audio, matching_audio = matching_audio, shifted_query_f0 = shifted_query_f0, query_f0 = query_f0, shifted_matching_f0 = shifted_matching_f0, matching_f0 = matching_f0, topk=512, is_going_up = (matching_f0_median > query_f0_median))


# out_wav = knn_vc.match(query_seq, matching_set, topk=4)
def play_sequence(audio_chunk, f_s = 16000):
	import sounddevice as sd
	sd.play(audio_chunk, f_s, blocking = True)
	
play_sequence(out_wav.cpu().numpy())
import sys
sys.exit()
# out_wav is (T,) tensor converted 16kHz output wav using k=4 for kNN.
# write_audio("/home/ken/Downloads/" + os.path.basename(src_wav_path).replace(".wav", "_knn_converted.wav"), out_wav.cpu().numpy(), sample_rate = 16000)

# VAD not in place!!!
import os
src_identifier = os.path.basename(src_wav_path).split(".")[0]
ref_identifier = os.path.basename(ref_wav_paths[0]).split(".")[0]
write_audio("/home/ken/Downloads/" + src_identifier + "_to_" + ref_identifier + "_knn_converted.wav", out_wav.cpu().numpy(), sample_rate = 16000)


# python prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/train/ --out_path /home/ken/Downloads/knn_vc_data/cached/train/ --topk 4 --matching_layer 6 --synthesis_layer 6 --prematch

# python prematch_dataset.py --librispeech_path /home/ken/Downloads/knn_vc_data/valid/ --out_path /home/ken/Downloads/knn_vc_data/cached/valid/ --topk 4 --matching_layer 6 --synthesis_layer 6 --prematch

# python -m hifigan.ddsp_train --audio_root_path_train /home/ken/Downloads/knn_vc_data/train --audio_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria/valid --feature_root_path_train /home/ken/Downloads/knn_vc_data/cached/train --feature_root_path_valid /home/ken/Downloads/knn_vc_data/Cantoria_cached/valid --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path /home/ken/Downloads/knn_vc_data/ckpt --fp16 False --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 1800 --fine_tuning

# python -m hifigan.train --audio_root_path /home/ken/Downloads/knn_vc_data/Cantoria --feature_root_path /home/ken/Downloads/knn_vc_data/Cantoria_cached --input_training_file data_splits/wavlm-hifigan-train.csv --input_validation_file data_splits/wavlm-hifigan-valid.csv --checkpoint_path /data2/home/ken/Downloads/knn_vc_data/ckpt --fp16 False --config hifigan/config_v1_wavlm.json --stdout_interval 25 --training_epochs 1800000 --fine_tuning

# python ./ddsp_inference.py ../../Downloads/temp_Choral_not_used/ctd_1_s_ans_resampled_16000.wav ../../Downloads/temp_Choral_not_used/ctd_1_b_ans_resampled_16000.wav no_post_opt mix
# python ./ddsp_inference.py ../../clips/matsuoka_yoshitsugu_resampled_16000.wav ../../clips/takahashi_rie_resampled_16000.wav no_post_opt mix


# VQMIVC
# python convert_example.py -s ../../Downloads/temp_Choral_not_used/ctd_1_b_ans_resampled_16000.wav   -r ../../Downloads/temp_Choral_not_used/ctd_1_t_ans_resampled_16000.wav -c converted -m /home/ken/Downloads/VQMIVC_data/checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpTrue/VQMIVC-model.ckpt-500.pt
