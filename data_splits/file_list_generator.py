import sys
src_dataset_path = sys.argv[1]
tgt_dataset_path = sys.argv[2]

import os
assert os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2])


from pathlib import Path
src_audio_files = list(Path(src_dataset_path).glob("**/*.wav")) + list(Path(src_dataset_path).glob("**/*.flac"))



src_spk_folders = list(set(audio_file.parent for audio_file in src_audio_files))

tgt_audio_files = list(Path(tgt_dataset_path).glob("**/*.wav")) + list(Path(tgt_dataset_path).glob("**/*.flac"))
tgt_spk_folders = list(set(audio_file.parent for audio_file in tgt_audio_files))


# for similarity, generate 
# src_speaker,tgt_speaker,x_path,y_path,label
# where x_path: uttr_name/tgt_name, y_path: some random tgt's uttr
sim_rows = []
intelli_rows = []


def level_one_up(src_file, src_root_dir = None, target_dir = None):
	
	from pathlib import Path
	src_file = Path(src_file)
	if src_root_dir is None:
		src_root_dir = src_file.parent.parent

	
	if target_dir is None:
		target_dir = src_root_dir
		
	import os
	rel_src_file = os.path.join(target_dir, os.path.relpath(src_file.parent.parent, src_root_dir), os.path.basename(src_file))
	
	return rel_src_file
	
	


for i, src_spk_folder in enumerate(src_spk_folders):
	
	
	src_audio_files = sorted(list(Path(src_spk_folder).glob("**/*.wav")) + list(Path(src_spk_folder).glob("**/*.flac")))
	
	'''
	src_audio_files = list(Path(src_spk_folder).glob("**/*"))
	for item in src_audio_files:
		# print(item, level_one_up(item, src_dataset_path, '/home/ken/Downloads/knn_vc_data/OpenSinger_test_flattened/'))
		# import sys
		# sys.exit()
		
		cp_to_file = level_one_up(item, src_dataset_path, '/home/ken/Downloads/knn_vc_data/OpenSinger_test/')
		os.system(f"mkdir -p {str(Path(cp_to_file).parent)}")
		os.system(f"cp {str(item)} {cp_to_file}")
	
	continue
	'''
	
	

	# wer/cer
	intelli_rows += [os.path.relpath(path, src_dataset_path) for path in src_audio_files[:300//len(src_spk_folders)]]


	from random import shuffle
	tgt_count = 0
	
	shuffle(tgt_spk_folders)
	for j, tgt_spk_folder in enumerate(tgt_spk_folders):
		tgt_audio_files = list(Path(tgt_spk_folder).glob("**/*.wav")) + list(Path(tgt_spk_folder).glob("**/*.flac"))
		
		if src_spk_folder == tgt_spk_folder:
			continue
			
		if tgt_count == 3:
			break
		tgt_count += 1
		
		'''
		chosen_indices = []
		while len(chosen_indices) < len(src_audio_files):
			remaining_to_collect = len(src_audio_files) - len(chosen_indices)
			
			import numpy
			temp_chosen_indices = np.random.choice(range(len(tgt_audio_files)), size=min(len(tgt_audio_files), remaining_to_collect), replace = False)
			chosen_indices += list(temp_chosen_indices)
		'''
		
		print(len(src_audio_files), len(tgt_audio_files))
		# assert len(chosen_indices) == len(src_audio_files)
		temp_gt_idx = 0
		temp_offset = 1
		# for each utterance, it should be converted to this tgt_spk
		for idx, src_audio_file in enumerate(src_audio_files):
			uttr_basename = ".".join(os.path.basename(src_audio_file).split(".")[:-1])
			tgt_spk = os.path.basename(tgt_spk_folder)
			src_spk = os.path.basename(src_spk_folder)
			
			# sim_rows.append([src_spk, tgt_spk, uttr_basename + "/" + tgt_spk, tgt_audio_files[chosen_indices[idx]], 0])
			sim_rows.append([src_spk, tgt_spk, uttr_basename + "/" + tgt_spk, ".".join(os.path.relpath(tgt_audio_files[temp_gt_idx], tgt_dataset_path).split(".")[:-1]), 0])
			
			if temp_gt_idx + temp_offset < len(tgt_audio_files):
				sim_rows.append([tgt_spk, tgt_spk, ".".join(os.path.relpath(tgt_audio_files[temp_gt_idx], tgt_dataset_path).split(".")[:-1]), ".".join(os.path.relpath(tgt_audio_files[temp_gt_idx + temp_offset], tgt_dataset_path).split(".")[:-1]), 1])
				
			else:
				sim_rows.append([tgt_spk, tgt_spk, ".".join(os.path.relpath(tgt_audio_files[temp_gt_idx], tgt_dataset_path).split(".")[:-1]), ".".join(os.path.relpath(tgt_audio_files[temp_gt_idx + temp_offset - len(tgt_audio_files)], tgt_dataset_path).split(".")[:-1]), 1])
			
				
			
			if temp_gt_idx == len(tgt_audio_files) - 1:
				temp_gt_idx = 0
				temp_offset += 1
			else:
				temp_gt_idx += 1
				

# import sys
# sys.exit()
	
print(len(list(Path(src_dataset_path).glob("**/*.wav")) + list(Path(src_dataset_path).glob("**/*.flac"))), len(intelli_rows), len(sim_rows))


output_folder = "/home/ken/open/knn-vc-master/data_splits/"
sim_output_file = output_folder + f"{os.path.basename(src_dataset_path.rstrip('/'))}_to_{os.path.basename(tgt_dataset_path)}.txt"
intelli_output_file = output_folder + f"{os.path.basename(src_dataset_path.rstrip('/'))}_intelli.txt"


intelli_input_file = output_folder + 'test-clean.py'
sim_input_file = output_folder + 'speaker-sim-test-clean.csv'


# for flatten the original LibriSpeech

intelli_rows = []
sim_rows = []
with open(intelli_input_file, "r") as f:
	for line in f:
		assert len(line.strip().split("/")) == 3, line.strip().split("/")
		intelli_rows.append(line.strip().split("/")[0] + "/" + line.strip().split("/")[-1])
		
with open(sim_input_file, "r") as f:
	for i, line in enumerate(f):
		if i == 0:
			continue
		
		tokens = line.strip().split(",")
		assert len(tokens[3].split("/")) == 3, tokens[3]
		assert tokens[-1] == "0" or tokens[-1] == "1"
		if tokens[-1] == "1":
			tokens[2] = tokens[2].split("/")[0] + "/" + tokens[2].split("/")[-1]
		tokens[3] = tokens[3].split("/")[0] + "/" + tokens[3].split("/")[-1]

		sim_rows.append(tokens)

# import sys
# sys.exit()




with open(intelli_output_file, "w") as f:
	for row in intelli_rows:
		f.write(str(row) + "\n")

with open(sim_output_file, "w") as f:
	f.write("src_speaker,tgt_speaker,x_path,y_path,label" + "\n")
	for row in sim_rows:
		f.write(",".join(str(item) for item in row) + "\n")


# python ./data_splits/file_list_generator.py ../../Downloads/knn_vc_data/test ../../Downloads/knn_vc_data/test	
# python ./data_splits/file_list_generator.py ../../Downloads/knn_vc_data/OpenSinger_test ../../Downloads/knn_vc_data/OpenSinger_test
# python ./data_splits/file_list_generator.py ../../Downloads/knn_vc_data/OpenSinger_test ../../Downloads/knn_vc_data/nus-smc-corpus_48
