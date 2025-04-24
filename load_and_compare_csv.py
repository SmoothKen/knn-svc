# not part of the training/inference scripts.


def csv_read(csv_file):
	import csv

	with open(csv_file, "r") as fp:
		reader = csv.reader(fp, delimiter=",", quotechar='"')
		# next(reader, None)  # skip the headers
		return [row for row in reader]

	
def csv_write(csv_file, data):
	with open(csv_file, "wt") as fp:
		writer = csv.writer(fp, delimiter=",")
		# writer.writerow(["your", "header", "foo"])  # write header
		writer.writerows(data)


import numpy as np
mix_rows = csv_read("/home/ken/open/knn-vc-master/speaker_sim_mix.csv")[1:]
wavlm_only_rows = csv_read("/home/ken/open/knn-vc-master/speaker_sim_wavlm_only.csv")[1:]

mix_scores = np.array([float(item[-2]) for item in mix_rows])
wavlm_only_scores = np.array([float(item[-2]) for item in wavlm_only_rows])

score_diff = mix_scores - wavlm_only_scores
worst_k_diff_idx = np.argsort(score_diff)[:5]
best_k_diff_idx = np.argsort(score_diff)[-5:]

# print(worst_k_diff_idx, score_diff[worst_k_diff_idx])
for idx in worst_k_diff_idx:
	print(idx, score_diff[idx], mix_rows[idx][3:5])

print("-----------")
# print(best_k_diff_idx, score_diff[best_k_diff_idx])
for idx in best_k_diff_idx:
	print(idx, score_diff[idx], mix_rows[idx][3:5])


