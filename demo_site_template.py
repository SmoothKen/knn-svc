
body_str = """
<!DOCTYPE html>
<html>

<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>knn-svc demo page</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
            font-size: 20px;
            line-height: 30px;
            margin: 40;
            padding: 40px 26px 26px 0px;
        }



		table {
            display: block;
            width: 100%;
            border-collapse: collapse;
            overflow: auto;
		}

		td, th {
		  border: 1px solid #dddddd;
		  text-align: left;
		  padding: 8px;
		}

		tr:nth-child(even) {
		  background-color: #dddddd;
		}
        

		h1 {
			text-align: center;
			display: block;
			font-size: 2em;
			line-height: 40px;
			font-weight: bold;
		}
        
    </style>
</head>
"""


def move_file_to_root(item_list, old_root, new_root):
	
	from pathlib import Path
	
	
	for item in item_list:
		if os.path.isfile(item):
			assert old_root in item
			tgt_path = os.path.join(new_root, os.path.relpath(old_root, item))
			print(tgt_path)
			import sys
			sys.exit()
			
			
			Path(tgt_path).parent.mkdir(parents = True, exist_ok = True)
			os.system(f"cp -u {item} {tgt_path}")
			


# two dimensional list, each item [0] -> content, [1] -> is_header
def list_to_html_table(item_list, is_header, num_cols, width = None):
	table_str = """
	<table>
		<tbody>
	"""
	
	# item_list = item_list[:num_cols]
	# is_header = is_header[:num_cols]
	
	cur_col_idx = 0
	assert len(item_list) % num_cols == 0 and len(item_list) == len(is_header), [len(item_list), num_cols]
	
	for item_idx, item in enumerate(item_list):
		
		if cur_col_idx == 0:
			table_str += "<tr>"
		elif cur_col_idx % num_cols == 0:
			table_str += "</tr><tr>"
			
		import os
		if os.path.isfile(item):
			if width is None:
				item = f"<audio controls preload src={item}></audio>"
			else:
				item = f"<audio controls style='width: {width}px;' preload src={item}></audio>"
				 
		
		
		# 0 -> content, 1 -> is_header
		if is_header[item_idx]:
			table_str += "<th>" + item + "</th>"
		else:
			table_str += "<td>" + item + "</td>"
		
		cur_col_idx += 1
	
	table_str += """
		</tr>
		</tbody>
	</table>






	"""
	
	return table_str
	
	


body_str += """
<h1>kNN-SVC: Robust Zero-Shot Singing Voice Conversion with Additive Synthesis and Concatenation Smoothness Optimization</h1>
"""

# Links: paper (arXiv PDF) and GitHub repo
body_str += """
<div style="text-align:center; margin: 10px 0 20px 0; font-size: 29px;">
    	<a href="https://arxiv.org/pdf/2504.05686" target="_blank" rel="noopener">Paper (arXiv PDF)</a>
    	&nbsp;|&nbsp;
    	<a href="https://github.com/SmoothKen/knn-svc" target="_blank" rel="noopener">Code (GitHub)</a>
    	&nbsp;|&nbsp;
		<a href="https://colab.research.google.com/github/SmoothKen/knn-svc/blob/master/knnsvc_demo.ipynb" target="_blank" rel="noopener">Demo (Colab)</a>
    	&nbsp;|&nbsp;
    	<a href="https://scholar.google.com/citations?user=jcQHdRgAAAAJ" target="_blank" rel="noopener">Keren's Google Scholar</a>
</div>
"""

body_str += """
<br><br>
<h2>Introduction</h2>
"""
body_str += """
<p>Let's start with a real use case. Say we want to convert the following Spanish bass to the following Spanish soprano. They have high phoneme overlaps due to being parts of a choral ensemble, creating a nice testing environment for zero-shot SVC.</p>
"""

table_00 = ["src", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000.wav", "ref", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_s_ans_resampled_16000.wav"]


body_str += list_to_html_table(table_00, [True]*len(table_00), 2, width = 800)

body_str += """
<br>
<p>Here are the conversion results. None of our baselines and ablations have been trained on Spanish audio.</p>
"""

table_01 = ["knn-svc w/o {AS, OPT}", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_no_harm_no_opt.wav", "NeuCoSVC", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_neuco_resampled_16000.wav", "knn-svc w/o {OPT}", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_harm_no_opt.wav", "knn-svc", "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_resampled_16000_knn_mix_harm_no_amp_0.552_post_opt_0.2.wav"]
# original final one "/home/ken/Downloads/temp_Choral_not_used/special_comparison/ctd_1_b_ans_resampled_16000_to_ctd_1_s_ans_harm_opt.wav"

body_str += list_to_html_table(table_01, [True]*len(table_01), 2, width = 800)


body_str += """
<ul>
  <li>knn-svc w/o {AS, OPT}: Sounds much duller than others, along with constant ringing.
  <li>NeuCoSVC: Sometimes with glitches (e.g. 00:07). Also, notice that the soprano sounds thinner, likely influenced by its Mandarin corpus (OpenSinger) training.</li>
  <li>knn-svc vs knn-svc w/o {OPT}: Notice that trembling (e.g. around 00:12, 00:26, 00:33, 00:51) significantly decreases. This is the main symptom the smoothness optimization intends to treat.</li>
</ul>  
"""
# 00:12

old_root = "/home/ken/Downloads/knn_vc_data"
new_root = "/home/ken/Downloads/knn_vc_data_samples"




body_str += """
<br><br>
<h2>Ablation and Model Comparisons</h2>
"""
body_str += """
<p>In this section, we provide samples for the ablation and model comparisons section (Table 1). "ls" refers to the LibriSpeech dataset. "os" refers to the OpenSinger dataset.</p>
"""

table_1 = ["", "src", "ref (not the entire pool)", "knn-vc", "knn-svc w/o {AS, OPT}", "NeuCoSVC",  "knn-svc w/o {OPT}", "knn-svc"]
num_cols = len(table_1)

##############################################
src_utts = ["8555/8555-284447-0023.flac", "908/908-31957-0014.flac", "1188/1188-133604-0009.flac"]
converted_utts = ["8555-284447-0023/908.flac", "908-31957-0014/4446.flac", "1188-133604-0009/121.flac"]
ref_utts = ["908/908-157963-0023.flac", "4446/4446-2275-0045.flac", "121/121-127105-0017.flac"]

ablation_folders = list("/home/ken/Downloads/knn_vc_data/" + item for item in ["test_converted_audio_wavlm_only_original/","test_converted_audio_mix_no_harm_no_amp_0.673/","test_converted_audio_mix_harm_no_amp_0.634_post_opt_False/","test_converted_audio_mix_harm_no_amp_0.634_post_opt_0.2/"])
# "test_converted_audio_mix_harm_no_amp_0.634_post_opt_True/"

gt_folder = "/home/ken/Downloads/knn_vc_data/test/"

import os
for utt_idx, utt in enumerate(converted_utts):
	table_1 += ["ls → ls"] + [os.path.join(gt_folder, item) for item in [src_utts[utt_idx], ref_utts[utt_idx]]] + [os.path.join(ablation_folder, converted_utts[utt_idx]) for ablation_folder in ablation_folders[:2]] + ["--"] + [os.path.join(ablation_folder, converted_utts[utt_idx]) for ablation_folder in ablation_folders[2:]]


##############################################

src_utts = ["ManRaw_26/26_拯救_42.wav", "WomanRaw_47/47_马德里不思议_44.wav"]
converted_utts = ["26_拯救_42/WomanRaw_47.wav", "47_马德里不思议_44/ManRaw_27.wav"]
ref_utts = ["WomanRaw_47/47_看得最远的地方_38.wav", "ManRaw_27/27_那个男人_49.wav"]

ablation_folders = list("/home/ken/Downloads/knn_vc_data/" + item for item in [ "OpenSinger_test_converted_audio_mix_no_harm_no_amp_0.636", "OpenSinger_test_converted_audio_neuco","OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_False", "OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_0.2"])
# "OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True"


gt_folder = "/home/ken/Downloads/knn_vc_data/OpenSinger_test/"

for utt_idx, utt in enumerate(converted_utts):
	table_1 += ["os → os"] + [os.path.join(gt_folder, item) for item in [src_utts[utt_idx], ref_utts[utt_idx]]] + ["--"] + [os.path.join(ablation_folder, converted_utts[utt_idx]) for ablation_folder in ablation_folders]

##############################################


src_utts = ["WomanRaw_47/47_遇见_9.wav", "ManRaw_26/26_征服_31.wav"]
converted_utts = ["47_遇见_9/JLEE.wav", "26_征服_31/MCUR.wav"]
ref_utts = ["JLEE//15_001.wav",  "MCUR/17_010.wav"]

ablation_folders = list("/home/ken/Downloads/knn_vc_data/" + item for item in [ "OpenSinger_test_to_nus-smc-corpus_48_audio_mix_no_harm_no_amp_0.636/","OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/","OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_False/", "OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_0.2/"])
# "OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/"

gt_folder_os = "/home/ken/Downloads/knn_vc_data/OpenSinger_test/"
gt_folder = "/home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/"

for utt_idx, utt in enumerate(converted_utts):
	table_1 += ["os → nus48e"] + [os.path.join(gt_folder_os, src_utts[utt_idx]), os.path.join(gt_folder, ref_utts[utt_idx])] + ["--"] + [os.path.join(ablation_folder, converted_utts[utt_idx]) for ablation_folder in ablation_folders]



body_str += list_to_html_table(table_1, [True]*len(table_1), num_cols)
# move_file_to_root(table_1, "/home/ken/Downloads/knn_vc_data", "/home/ken/Downloads/knn_vc_data_samples")


body_str += """
<br><br>
<h2>Duration Study</h2>
"""
body_str += """
<p>In this section, we provide examples for the duration study section (not in the paper due to length limit). The durations indicate those of the references.</p>
"""


##############################################
duration_strs = ["duration_limit_5_", "duration_limit_10_", "duration_limit_30_", "duration_limit_60_", "duration_limit_90_", ""]

src_utts = ["ManRaw_26/26_拯救_42.wav", "WomanRaw_47/47_马德里不思议_44.wav"]
converted_utts = ["26_拯救_42/WomanRaw_47.wav", "47_马德里不思议_44/ManRaw_27.wav"]
ref_utts = ["WomanRaw_47/47_看得最远的地方_38.wav", "ManRaw_27/27_那个男人_49.wav"]
ablation_folders = list("/home/ken/Downloads/knn_vc_data/" + item for item in ["OpenSinger_test_converted_audio_neuco", "OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_0.2"])
# "OpenSinger_test_converted_audio_mix_harm_no_amp_0.552_post_opt_True"

gt_folder = "/home/ken/Downloads/knn_vc_data/OpenSinger_test/"

table_2 = ["", "src", "ref (not the entire pool)", "5s", "10s", "30s", "60s", "90s", "full(~600s)"]
num_cols = len(table_2)

for utt_idx, utt in enumerate(converted_utts):
	table_2 += ["NeuCoSVC os → os"] + [os.path.join(gt_folder, item) for item in [src_utts[utt_idx], ref_utts[utt_idx]]] + [os.path.join(ablation_folders[0].replace("knn_vc_data/", f"knn_vc_data/{duration_str}"), converted_utts[utt_idx]) for duration_str in duration_strs]
	table_2 += ["knn-svc os → os"] + [os.path.join(gt_folder, item) for item in [src_utts[utt_idx], ref_utts[utt_idx]]] + [os.path.join(ablation_folders[1].replace("knn_vc_data/", f"knn_vc_data/{duration_str}"), converted_utts[utt_idx]) for duration_str in duration_strs]


##############################################

src_utts = ["WomanRaw_47/47_遇见_9.wav", "ManRaw_26/26_征服_31.wav"]
converted_utts = ["47_遇见_9/JLEE.wav", "26_征服_31/MCUR.wav"]
ref_utts = ["JLEE//15_001.wav",  "MCUR/17_010.wav"]

ablation_folders = list("/home/ken/Downloads/knn_vc_data/" + item for item in [ "OpenSinger_test_to_nus-smc-corpus_48_audio_neuco/", "OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_0.2/"])
# "OpenSinger_test_to_nus-smc-corpus_48_audio_mix_harm_no_amp_0.552_post_opt_True/"

gt_folder = "/home/ken/Downloads/knn_vc_data/nus-smc-corpus_48/"


for utt_idx, utt in enumerate(converted_utts):
	table_2 += ["NeuCoSVC os → nus"] + [os.path.join(gt_folder_os, src_utts[utt_idx]), os.path.join(gt_folder, ref_utts[utt_idx])] + [os.path.join(ablation_folders[0].replace("knn_vc_data/", f"knn_vc_data/{duration_str}"), converted_utts[utt_idx]) for duration_str in duration_strs]
	table_2 += ["knn-svc os → nus"] + [os.path.join(gt_folder_os, src_utts[utt_idx]), os.path.join(gt_folder, ref_utts[utt_idx])] + [os.path.join(ablation_folders[1].replace("knn_vc_data/", f"knn_vc_data/{duration_str}"), converted_utts[utt_idx]) for duration_str in duration_strs]


body_str += list_to_html_table(table_2, [True]*len(table_2), num_cols)
# move_file_to_root(table_2, "/home/ken/Downloads/knn_vc_data", "/home/ken/Downloads/knn_vc_data_samples")



body_str += """
<br><br>
<h2>Bonus</h2>
"""
body_str += """
<p>In this section, we play with knn-svc in various real use cases.</p>
"""

# "knn-svc bare converted",
# "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000_to_Hillary_Clinton_voice_bare.wav",
table_5 = ["src (English Male Speaker)", "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000.wav",
"ref (English Female Speaker)",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/Hillary_Clinton_voice_resampled_16000.wav", 
"knn-vc converted",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000_to_Hillary_Clinton_voice_knn_vc.wav",
"knn-svc converted",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000_to_Hillary_Clinton_voice_resampled_16000_knn_mix_harm_no_amp_0.633_post_opt_0.2.wav"]
# "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/we_will_make_america_great_again_resampled_16000_to_Hillary_Clinton_voice_knn_svc.wav"


num_cols = 2

body_str += list_to_html_table(table_5, [True]*len(table_5), num_cols, width = 800)
body_str += "<br>"



table_3 = ["src (French Male Singer)", "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut.wav",
"ref (English Male Singer (with guitar))",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/Tiken_lead_07_resampled_16000_cut.wav", 
"knn-svc converted", "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut_to_Tiken_lead_07_resampled_16000_cut_knn_mix_harm_no_amp_0.552_post_opt_0.wav"]
# "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/Danakil-voice_resampled_16000_cut_to_Tiken_lead_07_harm_opt.wav"

num_cols = 2

body_str += list_to_html_table(table_3, [True]*len(table_3), num_cols, width = 800)
body_str += "<br>"

table_4 = ["src (Japanese Male Singer (with vocal extraction))", "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/1_fuyu_no_hana_8k_Vocals_resampled_16000.wav",
"ref (Japanese Female Singer (with vocal extraction))",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/1_idol_yoasobi_Vocals_resampled_16000.wav", 
"knn-svc converted",
"/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/1_fuyu_no_hana_8k_Vocals_resampled_16000_to_1_idol_yoasobi_Vocals_resampled_16000_knn_mix_harm_no_amp_0.552_post_opt_0.2.wav"]
# "/home/ken/Downloads/temp_Choral_not_used/extra_comparisons/1_fuyu_no_hana_8k_Vocals_resampled_16000_to_1_idol_yoasobi_Vocals_resampled_16000_knn_converted.wav"

num_cols = 2

body_str += list_to_html_table(table_4, [True]*len(table_4), num_cols, width = 800)



body_str += """
</body>

</html>
"""

with open("/home/ken/Downloads/index.html", "w") as f:
	f.write(body_str)


'''
with open("/home/ken/Downloads/extra_file", "w") as f:
	for item in table_00 + table_01 + table_1 + table_2:
		if os.path.isfile(item):
			f.write(item + "\n")
'''

# 
inp = input('SERVER')
os.system(f"rsync -rvzu /home/ken/Downloads/index.html root@{inp}:/var/www/html/")


import sys
sys.exit()


# os.system(f"rsync --relative -rvz /home/ken/Downloads/temp_Choral_not_used/special_comparison/ root@{inp}:/var/www/html/")
# os.system(f"rsync --relative -rvz /home/ken/Downloads/temp_Choral_not_used/extra_comparisons/ root@{inp}:/var/www/html/")

for item in table_3 + table_4 + table_5 + table_00 + table_01 + table_1 + table_2:
	if os.path.isfile(item):
		from pathlib import Path
		# srv/http/
		os.system(f"rsync --relative -rvzu {item} root@{inp}:/var/www/html/")
		

