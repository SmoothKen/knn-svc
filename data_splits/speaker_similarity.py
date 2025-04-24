import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

import torchaudio
import torchaudio.functional as AF
from speechbrain.pretrained import EncoderClassifier
import pandas

from tqdm import tqdm


def eer(y, y_score):
	fpr, tpr, _ = roc_curve(y, 1 - y_score, pos_label=1)
	return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_speaker_similarity(args):
	classifier = EncoderClassifier.from_hparams(
		source="speechbrain/spkrec-xvect-voxceleb",
		savedir="pretrained_models/spkrec-xvect-voxceleb",
		run_opts={"device": "cuda"},
	)

	pairs = pandas.read_csv(args.eval_set)
	converted_pairs = pairs[pairs.label == 0]
	ground_truth_paris = pairs[pairs.label == 1]

	# essentially, those with label == 0 have x_path a spk1_to_spk2 utterance and y_path a (different) spk2 utterance
	# those with label == 1 have x_path a spk2 utterance and y_path a different spk2 utterance
	# the converted names are of the form utterance_name/tgt_speaker_id


	scores = []
	for _, (src, tgt, x_path, y_path, label) in tqdm(
		list(converted_pairs.iterrows()),
	):
		short_x_path = str(x_path).split("/")[0]
		short_y_path = str(y_path).split("/")[-1]
		# continue
		x_path = args.converted_dir / x_path
		y_path = args.ground_truth_dir / y_path
		# print(x_path)
		# import sys
		# sys.exit()
		
		
		import os
		if os.path.isfile(x_path.with_suffix(".flac")):
			x, sr = torchaudio.load(x_path.with_suffix(".flac"))
			y, sr = torchaudio.load(y_path.with_suffix(".flac"))
		elif os.path.isfile(x_path.with_suffix(".wav")):
			x, sr = torchaudio.load(x_path.with_suffix(".wav"))
			y, sr = torchaudio.load(y_path.with_suffix(".wav")) 
		else:
			print(x_path, y_path)
			raise NotImplementedError


		# x, sr = torchaudio.load(x_path.with_suffix(".wav"))
		
		# print(y_path, "/".join(str(y_path).split("/")[:-2] + [str(y_path).split("/")[-1]]))
		# import sys
		# sys.exit()
		
		# y, sr = torchaudio.load(Path("/".join(str(y_path).split("/")[:-2] + [str(y_path).split("/")[-1]])).with_suffix(".flac"))
		x = AF.resample(x, sr, 16000)
		y = AF.resample(y, sr, 16000)
		x, y = x.cuda(), y.cuda()

		x = classifier.encode_batch(x).squeeze().cpu().numpy()
		y = classifier.encode_batch(y).squeeze().cpu().numpy()

		'''
		if len(x.shape) > 1 or len(y.shape) > 1:
			print(x.shape, y.shape, y_path.with_suffix(".wav"))
			y, sr = torchaudio.load(y_path.with_suffix(".wav"))
			print(y.shape)
			import sys
			sys.exit()
		'''

		scores.append((src, tgt, short_x_path, short_y_path, cosine(x, y), label))
		# break

	for _, (src, tgt, x_path, y_path, label) in tqdm(
		list(ground_truth_paris.iterrows()),
	):
		
		short_x_path = str(x_path).split("/")[-1]
		short_y_path = str(y_path).split("/")[-1]
		
		x_path = args.ground_truth_dir / x_path
		y_path = args.ground_truth_dir / y_path
		# print(x_path, y_path)

		import os
		if os.path.isfile(x_path.with_suffix(".flac")):
			x, sr = torchaudio.load(x_path.with_suffix(".flac"))
			y, sr = torchaudio.load(y_path.with_suffix(".flac"))
		elif os.path.isfile(x_path.with_suffix(".wav")):
			x, sr = torchaudio.load(x_path.with_suffix(".wav"))
			y, sr = torchaudio.load(y_path.with_suffix(".wav"))
		else:
			raise NotImplementedError


		
		# x, sr = torchaudio.load(Path("/".join(str(x_path).split("/")[:-2] + [str(x_path).split("/")[-1]])).with_suffix(".flac"))
		# y, sr = torchaudio.load(Path("/".join(str(y_path).split("/")[:-2] + [str(y_path).split("/")[-1]])).with_suffix(".flac"))
		
		
		x = AF.resample(x, sr, 16000)
		y = AF.resample(y, sr, 16000)
		x, y = x.cuda(), y.cuda()

		x = classifier.encode_batch(x).squeeze().cpu().numpy()
		y = classifier.encode_batch(y).squeeze().cpu().numpy()
		
		# import sys
		# sys.exit()

		scores.append((src, tgt, short_x_path, short_y_path, cosine(x, y), label))
		# break


	scores = pandas.DataFrame(
		scores, columns=["src_speaker", "tgt_speaker", "src_path", "tgt_path", "score", "label"]
	)
	sim = (
		scores.groupby("tgt_speaker")
		.apply(lambda x: eer(x.label, x.score))
		.reset_index(name="eer")
	)
	
	import os
	from pathlib import Path
	script_path = Path(os.path.realpath(__file__)).parent.parent
	
	scores.to_csv(script_path / f"{os.path.basename(args.converted_dir)}_sim_result.txt")
	# import sys
	# sys.exit()
	
	return sim.agg(mean=("eer", np.mean), std=("eer", np.std))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate speaker similarity.")
	parser.add_argument(
		"eval_set",
		metavar="eval-set",
		help="path to csv listing the test pairs.",
		type=Path,
	)
	parser.add_argument(
		"converted_dir",
		metavar="converted-dir",
		help="path to the directory containing the converted speech.",
		type=Path,
	)
	parser.add_argument(
		"ground_truth_dir",
		metavar="ground-truth-dir",
		help="path to the directory containing the ground-truth speech.",
		type=Path,
	)
	args = parser.parse_args()
	sim = compute_speaker_similarity(args)
	print(sim)

# python -m pip install speechbrain
# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/speaker-sim-test-clean.csv /home/ken/Downloads/knn_vc_data/test_converted_audio/ /home/ken/Downloads/knn_vc_data/test/
# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio/ /home/ken/Downloads/knn_vc_data/test/


# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio_wavlm_only_original/ /home/ken/Downloads/knn_vc_data/test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_no_harm_no_amp_0.673/ /home/ken/Downloads/knn_vc_data/test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_False/ /home/ken/Downloads/knn_vc_data/test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_True/ /home/ken/Downloads/knn_vc_data/test/

# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/test_to_test.txt /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_extra/ /home/ken/Downloads/knn_vc_data/test/


# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_mix_no_harm_no_amp_0.636/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_False/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;

# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_extra/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;



# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_mix_no_harm_no_amp_0.636/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_False/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; 

# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_extra/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/;





# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_5_OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;  python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_10_OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;  python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_30_OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;  python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_60_OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;  python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_90_OpenSinger_test_converted_audio_neuco/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; 

# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_5_OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_10_OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_30_OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_60_OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_OpenSinger_test.txt /home/ken/Downloads/knn_vc_data/duration_limit_90_OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/OpenSinger_test/;







# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_5_OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_10_OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_30_OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_60_OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_90_OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; 



# python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_5_OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_10_OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_30_OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_60_OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/; python ./data_splits/speaker_similarity.py /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_to_nus-smc-corpus_48.txt /home/ken/Downloads/knn_vc_data/duration_limit_90_OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/ /home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/;

