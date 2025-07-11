
import sys
sys.path.append("/home/ken/open")
from lib_ongaku_test import save_audio

import torch, torchaudio

# knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
import sys
if len(sys.argv) < 3:
	print("Enter content wav as \$1, and style wavs as \$2...")
	






# WavLM ckpt: /home/ken/.cache/torch/hub/checkpoints/WavLM-Large.pt
# HiFi-GAN ckpt: /home/ken/.cache/torch/hub/checkpoints/prematch_g_02500000.pt

# Or, if you would like the vocoder trained not using prematched data, set prematched=False.


topk = 4
device = "cuda"
prioritize_f0 = True
tgt_loudness_db = -16



import os
# for now, args are SRC_FILE, TGT_FILE, POST_OPT_TYPE, CKPT_TYPE
# if the first path is a file, then assume the simplest src file to ref file case
if os.path.isfile(sys.argv[1]):
	assert len(sys.argv) > 4
	
	from ddsp_hubconf import knn_vc
	knn_vc = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda', ckpt_type = sys.argv[-1])
	# src_wav_path = sys.argv[1]
	# ref_wav_paths = sys.argv[2:-1]

	
	src_wav_path = sys.argv[1]
	ref_wav_path = sys.argv[2]

	out_wav = knn_vc.special_match(src_wav_file = src_wav_path, ref_wav_file = ref_wav_path, topk = topk, device = device, prioritize_f0 = prioritize_f0, ckpt_type = sys.argv[-1], tgt_loudness_db = tgt_loudness_db, post_opt = sys.argv[-2])

	import sys
	sys.exit()


# import argparse
# parser = argparse.ArgumentParser()

# "/home/ken/Downloads/knn_vc_data/test"
# "/home/ken/Downloads/knn_vc_data/OpenSinger_test"
# "/home/ken/Downloads/knn_vc_data/OpenSinger_test"
# parser.add_argument('--src_folder', type=str)
# "/home/ken/Downloads/knn_vc_data/test"
# "/home/ken/Downloads/knn_vc_data/OpenSinger_test"
# "/home/ken/Downloads/knn_vc_data/nus-smc-corpus_48"
# parser.add_argument('--tgt_folder', type=str)
# parser.add_argument('--ckpt_type', type=str)
# parser.add_argument('--post_opt', type=str)


# parser.add_argument('--topk', default=4)
# parser.add_argument('--device', default="cuda")
# parser.add_argument('--prioritize_f0', default=True)
# parser.add_argument('--tgt_loudness_db', default=-16)


# "/home/ken/open/knn-vc-master/data_splits/test_to_test.txt"
# "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt"
# "/home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt"
parser.add_argument('--required_subset_file', default=None)
parser.add_argument('--with_dur_limit', default=False)
args = parser.parse_args()





assert any(item in arg.ckpt_type for item in {"wavlm_only", "mix"}), "Bad sys.argv[-1]"


# "wavlm_only" -> the original knn-vc, 
# mix -> some relic
# mix_no_harm_no_amp, bare pitch guidance
# mix_harm_no_amp -> with AS
if "wavlm_only" in arg.ckpt_type or "no_harm_no_amp" in arg.ckpt_type:
	arg.post_opt = "no_post_opt"
else:
	assert arg.post_opt.startswith("post_opt") or arg.post_opt.startswith("no_post_opt")



from ddsp_hubconf import knn_vc
knn_vc = knn_vc(pretrained=True, progress=True, prematched=True, device='cuda', ckpt_type = args.ckpt_type)

if args.with_dur_limit:
	duration_limits = ["5", "10", "30", "60", "90"]
else:
	duration_limits = [None]

	
for duration_limit in duration_limits:
	
	converted_audio_dir = f"/home/ken/Downloads/knn_vc_data/{os.path.basename(arg.src_folder)}_to_{os.path.basename(arg.tgt_folder)}_" + arg.ckpt_type + f"_post_opt_{arg.post_opt}/"
	
	if durtion_limit is not None:
		converted_audio_dir = f"duration_limit_{duration_limit}_" + converted_audio_dir
	
	knn_vc.bulk_match(src_dataset_path = arg.src_folder,  tgt_dataset_path = arg.tgt_folder, converted_audio_dir = converted_audio_dir, topk = arg.topk, device = arg.device, prioritize_f0 = arg.prioritize_f0, ckpt_type = arg.ckpt_type, tgt_loudness_db = arg.tgt_loudness_db, required_subset_file = arg.required_subset_file, post_opt = arg.post_opt, duration_limit = duration_limit)









# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut.wav ../../Downloads/temp_Choral_not_used/extra_comparisons/Tiken_lead_07_resampled_16000_cut.wav no_post_opt mix
# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut.wav ../../Downloads/temp_Choral_not_used/extra_comparisons/Tiken_lead_07_resampled_16000_cut.wav post_opt_0.2 mix

# % temp_plot command
# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/ctd_1_b_ans_resampled_16000.wav ../../Downloads/temp_Choral_not_used/ctd_1_s_ans_resampled_16000.wav post_opt_0.2 mix



# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/ctd_1_s_ans_resampled_16000.wav ../../Downloads/temp_Choral_not_used/ctd_1_b_ans_resampled_16000.wav no_post_opt mix_no_harm_no_amp_0.636




# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut.wav ../../Downloads/temp_Choral_not_used/extra_comparisons/Tiken_lead_07_resampled_16000_cut.wav post_opt_0.2 mix_harm_no_amp_0.552

# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/ctd_1_b_ans_resampled_16000.wav ../../Downloads/temp_Choral_not_used/ctd_1_s_ans_resampled_16000.wav post_opt_0.2 mix_harm_no_amp_0.552

# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000.wav ../../Downloads/temp_Choral_not_used/extra_comparisons/Hillary_Clinton_voice_resampled_16000.wav post_opt_0.2 mix_harm_no_amp_0.633


# python ./ddsp_inference_cleaned.py ../../Downloads/temp_Choral_not_used/extra_comparisons/1_fuyu_no_hana_8k_Vocals_resampled_16000.wav ../../Downloads/temp_Choral_not_used/extra_comparisons/1_idol_yoasobi_Vocals_resampled_16000.wav post_opt_0.2 mix_harm_no_amp_0.552


# python ./ddsp_inference.py ../../clips/matsuoka_yoshitsugu_resampled_16000.wav ../../clips/takahashi_rie_resampled_16000.wav no_post_opt mix

