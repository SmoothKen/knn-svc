
import sys
sys.path.append("/home/ken/open")

"""
Clean CLI for ddsp_inference with two modes:
  a) file-to-file conversion (content file -> style file)
  b) folder-to-folder bulk conversion (content folder -> style folder)

Defaults follow the previously hard-coded values to maintain behavior.
"""

import os
import argparse


def str2bool(v: str) -> bool:
	v = v.lower()
	if v in ("yes", "true", "t", "1", "y"): return True
	if v in ("no", "false", "f", "0", "n"): return False
	raise argparse.ArgumentTypeError("boolean value expected")


def main():
	parser = argparse.ArgumentParser(description="kNN-SVC inference: file or folder mode")
	# Positional paths; each can be a file or a folder
	_ = parser.add_argument("src", help="Content source: path to an audio file OR a dataset root (folder of speaker folders) of audio files grouped by speaker")
	_ = parser.add_argument("tgt", help="Style target: path to an audio file OR a dataset root (folder of speaker folders) of audio files grouped by speaker")

	# Model/runtime options (defaults replicate previous hard-coded values)
	_ = parser.add_argument("--ckpt_dir", type=str, default="/home/ken/Downloads/knn_vc_data/ckpt_saved", help="The directory in which the checkpoints are stored, if they are local ones")
	_ = parser.add_argument("--ckpt_type", type=str, default="mix", help="Checkpoint type: e.g., mix, mix_harm_no_amp_*, mix_no_harm_no_amp_*, wavlm_only, wavlm_only_original")

	_ = parser.add_argument("--post_opt", type=str, default="no_post_opt", help="inference-time smoothness optimization setting: e.g., no_post_opt or post_opt_0.2")


	_ = parser.add_argument("--required_subset_file", type=str, default=None, help="CSV defining subset for bulk conversion; if provided, restricts processed files")


	_ = parser.add_argument("--topk", type=int, default=4)
	_ = parser.add_argument("--device", type=str, default="cuda")
	_ = parser.add_argument("--prioritize_f0", type=str2bool, default=True)
	_ = parser.add_argument("--tgt_loudness_db", type=float, default=-16)


	_ = parser.add_argument("--dur_limit", type=int, default=None, help="Duration limit set on the target pool (i.e. will restrict to only the first N minutes)")


	args = parser.parse_args()

	# Load model
	from ddsp_hubconf import knn_vc
	knn = knn_vc(pretrained=True, progress=True, prematched=True, device=args.device, ckpt_type=args.ckpt_type, local_ckpt_dir = args.ckpt_dir)

	# Decide mode by filesystem
	src_is_file = os.path.isfile(args.src)
	tgt_is_file = os.path.isfile(args.tgt)
	src_is_dir = os.path.isdir(args.src)
	tgt_is_dir = os.path.isdir(args.tgt)

	if src_is_file and tgt_is_file:
		# Single file -> single file
		_ = knn.special_match(
			src_wav_file=args.src,
			ref_wav_file=args.tgt,
			topk=args.topk,
			device=args.device,
			prioritize_f0=args.prioritize_f0,
			ckpt_type=args.ckpt_type,
			tgt_loudness_db=args.tgt_loudness_db,
			post_opt=args.post_opt,
		)
		# special_match will save audio and may exit internally
		return

	if src_is_dir and tgt_is_dir:
		# Bulk folder -> folder
		duration_limits = [args.dur_limit]

		for duration_limit in duration_limits:
			converted_audio_dir = (
				f"/home/ken/Downloads/knn_vc_data/"
				f"{os.path.basename(args.src)}_to_{os.path.basename(args.tgt)}_"
				f"{args.ckpt_type}_post_opt_{args.post_opt}/"
			)
			if duration_limit is not None:
				converted_audio_dir = f"duration_limit_{duration_limit}_" + converted_audio_dir

			_ = knn.bulk_match(
				src_dataset_path=args.src,
				tgt_dataset_path=args.tgt,
				converted_audio_dir=converted_audio_dir,
				topk=args.topk,
				device=args.device,
				prioritize_f0=args.prioritize_f0,
				ckpt_type=args.ckpt_type,
				tgt_loudness_db=args.tgt_loudness_db,
				required_subset_file=args.required_subset_file,
				post_opt=args.post_opt,
				duration_limit=duration_limit,
			)
		return

	raise SystemExit("Both inputs must be files or both must be folders.")


if __name__ == "__main__":
	main()


# python ./ddsp_inference.py ../../clips/matsuoka_yoshitsugu_resampled_16000.wav ../../clips/takahashi_rie_resampled_16000.wav no_post_opt mix

