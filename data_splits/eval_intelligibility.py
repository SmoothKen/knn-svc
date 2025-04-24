import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastprogress.fastprogress import master_bar, progress_bar
from torch import Tensor
import re
from num2words import num2words
import jiwer
import whisper

WHISPER_DECODE_ARGS = {
	'verbose': None,
	'temperature': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
	'compression_ratio_threshold': 2.4,
	'logprob_threshold': -0.8,
	'no_speech_threshold': 0.8,
	'best_of': 20, 
	'beam_size': 20, 
	'without_timestamps': True,
	'fp16': True
}


def numbers_to_words(text):
	def _conv_num(match):
		res = num2words(match.group())
		res = res.replace('-', ' ')
		return res
	return re.sub(r'\b\d+\b', _conv_num, text)


def make_librispeech_df(root_path: Path) -> pd.DataFrame:
	all_files = list(root_path.rglob('**/*.flac')) + list(root_path.rglob('**/*.wav'))
	speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
	subset = [f.parents[2].stem for f in all_files]
	df = pd.DataFrame({'path': all_files, 'speaker': speakers, 'subset': subset})
	return df

def get_transcriptions(df: pd.DataFrame) -> pd.DataFrame:
	transcripts = {}
	out_transcripts = []
	for i, row in progress_bar(df.iterrows(), total=len(df)):
		p = Path(row.path)
		if p.stem in transcripts:
			out_transcripts.append(transcripts[p.stem])
		else:

			# with open(p.parent/f'{p.parents[1].stem}-{p.parents[0].stem}.trans.txt', 'r') as file:
			with open(p.parent/f'{"-".join(str(p).split("-")[:2])}.trans.txt', 'r') as file:
				lines = file.readlines()
				for l in lines:
					uttr_id, transcrip = l.split(' ', maxsplit=1)
					transcripts[uttr_id] = transcrip.strip()
			out_transcripts.append(transcripts[p.stem])
	df['transcription'] = out_transcripts
	return df



def get_transcriptions_other(df: pd.DataFrame) -> pd.DataFrame:
	transcripts = {}
	out_transcripts = []
	for i, row in progress_bar(df.iterrows(), total=len(df)):
		p = Path(row.path)

		assert os.path.isfile(str(p).replace(".wav", ".txt"))

		with open(str(p).replace(".wav", ".txt"), 'r') as file:
			lines = file.readlines()
			assert len(lines) == 1
			out_transcripts.append(lines[0].strip())

		# print(p, out_transcripts[-1])
		# import sys
		# sys.exit()
			
	df['transcription'] = out_transcripts
	
	return df


# for each utterance in librispeech, find the converted ones that uses the same utterance, transcribe them and compare

def main(args: argparse.Namespace):
	WHISPER_DECODE_ARGS['beam_size'] = args.beam
	WHISPER_DECODE_ARGS['best_of'] = args.beam
	
	# 1. build source transcriptions and utterances.
	ls_df = make_librispeech_df(Path(args.librispeech_path))
	if args.librispeech_path != "/home/ken/Downloads/knn_vc_data/test/":
		ls_df = get_transcriptions_other(ls_df)
	else:	
		ls_df = get_transcriptions(ls_df)


	# import sys
	# sys.exit()

	# load all audio, then filter to only those in source_uttr
	with open(args.source_uttrs, 'r') as file:
		items = file.readlines()
		items = [it.strip() for it in items]
	mask = [any([item in str(pth) for item in items]) for pth in ls_df['path'].tolist()]
	mask = np.array(mask)
	ls_df = ls_df[mask]
	
	# 2. load whisper 
	model = whisper.load_model(args.whisper, device=args.device)

	# pred_paths = list(Path(args.pred_path).rglob('**/*.wav'))
	pred_paths = list(Path(args.pred_path).rglob('**/*.flac')) + list(Path(args.pred_path).rglob('**/*.wav'))
	assert len(pred_paths) > 0, pred_paths
	
	assert WHISPER_DECODE_ARGS['beam_size'] == args.beam
	# 3. Loop through all source utterances
	gt_transcripts = []
	pred_transcripts = []
	mb = master_bar(ls_df.iterrows(), total=len(ls_df))
	for i, row in mb:
		
		cur_pred_paths = [p for p in pred_paths if p.parent.stem == Path(row.path).stem]
		
		# print(cur_pred_paths, row.path)
		# import sys
		# sys.exit()
		
		gt_transcript = row.transcription.strip().upper()
		pb = progress_bar(cur_pred_paths, parent=mb)
		# mb.write(f"Transcribing preds for source utterance {Path(row.path).stem} | subset = {row.subset}")
		# mb.write(f'GT: {row.transcription}')
		'''
		## ------------- whisper topline code --------------
		## NOTE: if you use this, ensure to comment out the next for loop.
		transcript = model.transcribe(str(row.path), language='english', **WHISPER_DECODE_ARGS)
		if transcript is list: transcript = transcript[0]
		pred_transcript = transcript['text'].strip().upper()
		gt_transcripts.append(gt_transcript)
		pred_transcripts.append(pred_transcript)
		print(" ", i, "/", len(ls_df), end="\r")
		# ------------- end whisper topline code ----------
		
		'''
		for cpath in pb:
			# NOTE: comment this out if you're just doing english
			if args.librispeech_path != "/home/ken/Downloads/knn_vc_data/test/":
				transcript = model.transcribe(str(cpath), language='mandarin', **WHISPER_DECODE_ARGS)
			else:
				transcript = model.transcribe(str(cpath), language='english', **WHISPER_DECODE_ARGS)
			# transcript = model.transcribe(str(cpath), language=str(row.subset).lower(), **WHISPER_DECODE_ARGS)
			if transcript is list: transcript = transcript[0]
			pred_transcript = transcript['text'].strip().upper()

			gt_transcripts.append(gt_transcript)
			pred_transcripts.append(pred_transcript)
			mb.child.comment = f"{pred_transcript[:5]}"
		
		
		
	# 3.post save cache, just in case. 
	# torch.save({'gt_transcripts': gt_transcripts, 'pred_transcripts': pred_transcripts}, './cache.pt')
	# 4. Normalize text
	# words to numbers (since whisper can be inconsistent)
	pred_transcripts = [numbers_to_words(p) for p in pred_transcripts]
	gt_transcripts = [numbers_to_words(p) for p in gt_transcripts]
	cer_text_cleaner = jiwer.Compose([
		jiwer.ToLowerCase(),
		jiwer.RemoveWhiteSpace(replace_by_space=True),
		jiwer.RemoveMultipleSpaces(),
		jiwer.RemovePunctuation(),
		jiwer.transforms.ReduceToListOfListOfChars()
	]) 

	wer_text_cleaner = jiwer.Compose([
		jiwer.ToLowerCase(),
		jiwer.RemoveWhiteSpace(replace_by_space=True),
		jiwer.RemoveMultipleSpaces(),
		jiwer.RemovePunctuation(),
		jiwer.transforms.ReduceToListOfListOfWords()
	]) 
	
	wer_measure = jiwer.compute_measures(gt_transcripts, pred_transcripts, 
						hypothesis_transform=wer_text_cleaner, 
						truth_transform=wer_text_cleaner)

	

	cer_measure = jiwer.compute_measures(gt_transcripts, pred_transcripts, 
						hypothesis_transform=cer_text_cleaner, 
						truth_transform=cer_text_cleaner)

	# print(type(wer_measure))
	print('-'*10 + ' WER metrics ' + '-'*10)
	print(wer_measure["wer"])
	print('-'*10 + ' CER metrics ' + '-'*10)
	print(cer_measure["wer"])

	import os
	with open(f'/home/ken/Downloads/{os.path.basename(args.pred_path)}_result.txt', 'w') as file:
		print(str(args.pred_path), file=file)
		print('\nWER measure\n', file=file)
		print(str(wer_measure), file=file)
		print('\nCER measure\n', file=file)
		print(str(cer_measure), file=file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Compute WER/CER using whisper.")

	parser.add_argument('--librispeech_path', required=True, type=str, help='path to librispeech subset directory. e.g. .../librispeech/test-clean/')
	parser.add_argument('--source_uttrs', required=True, type=str, help='path to list of source utterances')
	parser.add_argument('--seed', default=1776, type=int)
	parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'])
	parser.add_argument('--whisper', default='small', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'], help='Whisper ASR model size.')
	
	parser.add_argument('--beam', default=20, type=int, help='beam size for whisper evaluation')
	parser.add_argument('--pred_path', 
		required=True, type=str, 
		help='Path to output predicted .wav files. The parent of each file should correspond to the source utterance.'
	)

	parser.add_argument('--resume', action='store_true')

	args = parser.parse_args()
	main(args)

# 
# python -m pip install whisper num2words
# python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/test_converted_audio_wavlm_only_original; python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_no_harm_no_amp_0.673; python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_False; python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_True

# python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/test_converted_audio_mix_harm_no_amp_0.634_post_opt_extra





# python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/OpenSinger_test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/OpenSinger_test_converted/

# python ./data_splits/eval_intelligibility.py --librispeech_path /home/ken/Downloads/knn_vc_data/OpenSinger_test/ --source_uttrs /home/ken/open/knn-vc-master/data_splits/OpenSinger_test_intelli.txt --device cuda --pred_path /home/ken/Downloads/knn_vc_data/OpenSinger_test_to_nus48e_converted/
