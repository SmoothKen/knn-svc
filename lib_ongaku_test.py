def play_sequence(audio_chunk, f_s = 16000):
	import sounddevice as sd
	sd.play(audio_chunk, f_s, blocking = True)


def plot_matrix(mat, row_names = None, col_names = None, title = "", x_axis = "", y_axis = "", fig = None, fig_row = 2, fig_col = 1):

	import plotly.express as px
	show_on_screen = (fig is None)
	if show_on_screen:
		fig = px.imshow(mat, text_auto=True, x=col_names, y=row_names, aspect='auto', color_continuous_scale='Bluered_r')
		
		
	else:
		import plotly.graph_objects as go
		fig_imshow = px.imshow(mat, text_auto=True, x=col_names, y=row_names, aspect='auto', color_continuous_scale='Bluered_r')
		fig.add_trace(go.Heatmap(fig_imshow.data[0]),row=fig_row, col=fig_col)
		
		# fig.update_layout(coloraxis_showscale=True, coloraxis=dict(colorbar_len=0.5, colorbar_y=0.80))
		fig.update_layout(coloraxis_showscale=False)
	
	fig.update_layout(
		title=title,
		xaxis_title=x_axis,
		yaxis_title=y_axis,
		margin={"l":40, "r":40, "t":40, "b":40},
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		autosize=True,
		template="simple_white"
	)

	if show_on_screen:
		fig.show()	
	else:
		return fig


# ys list of y sequences
def plot_multi_sequences(x, ys, y_names, title = "", template="plotly", width = None, height = None, x_axis = None, y_axis = None, initial_visibility = True, fig = None, fig_row = 1, fig_col = 1):
	'''
	
	import pandas as pd
	data_df = pd.DataFrame(ys, index=y_names, columns=x).T
	
	import plotly.express as px
	# print(data_df)
	fig = px.line(data_df)

	'''
	
	import plotly.graph_objects as go

	# https://community.plotly.com/t/hovertemplate-does-not-show-name-property/36139/2
	
	show_on_screen = (fig is None)
	if show_on_screen:
		fig = go.Figure(data = [go.Scatter(x = x, y = ys[i], name = y_names[i], meta = [y_names[i]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>') for i in range(len(ys))])
	else:
		fig.append_trace(go.Scatter(x = x, y = ys[0], name = y_names[0], meta = [y_names[0]], hovertemplate = '%{meta}<br>x=%{x}<br>y=%{y}<extra></extra>'), row=fig_row, col=fig_col)
	
	
	fig.update_layout(
		title=title,
		font=dict(size=25),
		hoverlabel=dict(font_size=25),
		margin={"l":40, "r":40, "t":40, "b":40},
		autosize=True,
		template=template,
		width=width,
		height=height,
		xaxis_title=x_axis, 
		yaxis_title=y_axis
	)
	
	
	if not initial_visibility:
		fig.update_traces(visible = 'legendonly')
		
		
	if show_on_screen:
		fig.show(config = {'showTips':False})
	else:
		return fig
	



def save_audio(filename, waveform, sample_rate):
	
	# first convert as we may need it to become bytes later
	import torch
	if isinstance(waveform, torch.Tensor):
		waveform = waveform.detach().cpu().numpy()

	
	import numpy as np
	# print(waveform.shape, np.max(waveform), np.min(waveform))

	
	# convert to int32 if it is [-1, 1] float
	if waveform.dtype == np.float32 or waveform.dtype == np.float64:
		
		# ensure it is in [-1, 1]
		waveform_abs_max = np.max(np.abs(waveform))
		if waveform_abs_max > 1:
			waveform = waveform/waveform_abs_max
	
		
		
		waveform = waveform * (2 ** 31 - 1)   
		waveform = waveform.astype(np.int32)
	else:
		assert waveform.dtype == np.int32
		
		

	if filename.endswith(".wav"):
		import soundfile as sf
		sf.write(filename, waveform.T, samplerate = sample_rate, subtype = 'PCM_32')
		
	else:
		
		
		if waveform.ndim == 2:
			if waveform.shape[0] in {1, 2}:
				waveform = waveform.T
				
			channels = waveform.shape[1]
		elif waveform.ndim == 1:
			channels = 1
		else:
			import sys
			sys.exit("Bad audio array shape")
		
			
		
		from pydub import AudioSegment
		song = AudioSegment(waveform.tobytes(), frame_rate=sample_rate, sample_width=4, channels=channels)
		
		assert filename.split(".")[-1] in {"mp3", "flac"}
		
		song.export(filename, format=filename.split(".")[-1], bitrate="320k")




def fast_cosine_dist(source_feats_collection, matching_pool, increment = 20):
	import torch
	source_norms_collection = torch.norm(source_feats_collection, p=2, dim=-1)
	matching_norms = torch.norm(matching_pool, p=2, dim=-1)
	

	matching_start = 0
	dists_collection = []
	while matching_start < len(source_feats_collection):
		
		source_norms = source_norms_collection[matching_start:matching_start+increment]
		source_feats = source_feats_collection[matching_start:matching_start+increment]
		# print(source_norms.shape, source_feats.shape)
			
		dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
		dotprod /= 2
	
		dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
		if torch.sum(torch.isnan(dists)) > 0:
			print("containing nan")
			import sys
			sys.exit()
			
			
		dists_collection.append(dists)
		matching_start += increment
	
	return torch.cat(dists_collection, dim = 0)


# src_elements (num_ele, dim)
# return (num_src_ele, num_tgt_ele)
# retain_mask, matrix (num_src, num_tgt), 1 if want to retain
# topk -> highest nbrs, if None, then return unsorted
def knn_cosine_similarity(src_elements, tgt_elements, retain_mask = None, topk = 32):
	import torch
	dists = fast_cosine_dist(src_elements.half().float(), tgt_elements.half().float())

	assert dists.shape[0] == src_elements.shape[0] and dists.shape[1] == tgt_elements.shape[0]
		
	
		
	if retain_mask is not None:
		assert retain_mask.shape == dists.shape
		dists = dists + (1 - retain_mask)
	
	
	topk_sort = dists.topk(k=topk, dim=-1, largest=False)
	return topk_sort.indices, topk_sort.values



# semitone = 0, shift_engine = "librosa"
def batch_load_audio(wav_files, sr = None, mono = False):
	import os, librosa
	
	# either list of dir, list of file,

	wav_list = []
	sr_list = []
	
	for wav_file in wav_files:
		
		assert os.path.isfile(wav_file), [wav_file]
		
		x, original_sr = librosa.load(wav_file, sr=sr, mono=mono)
		if len(x.shape) == 1:
			x = x[None, :]

		
			
		wav_list.append(x)
		sr_list.append(original_sr)
	
	
	# sanity check, ensure all having the same sr
	sr_list = list(set(sr_list))
	assert len(sr_list) == 1, sr_list
	return wav_list, sr_list[0]
		
	'''

	if semitone != 0:	
		if shift_engine == "librosa":
			import librosa
			shifted_x = librosa.effects.pitch_shift(x, sr=sr, n_steps=semitone, bins_per_octave=12)
			
		else:
			x = torch.tensor(x)
			import torchaudio
			transform = transforms.PitchShift(sample_rate=sr, n_steps=int(semitone), bins_per_octave=12)
			shifted_x = transform(x)

	'''


# TODO: consider +0 when +1 in the reference does not produce long enough steadiness?
# +0 not a good idea, bad noise

# slice_list: [[start_idx, end_idx], ...] 
def smoothen_f0(f0, slice_list, frame_per_second = 50):
	
	import numpy as np
	for item in slice_list:
		start_idx = int(item[0]*frame_per_second)
		end_idx = int(item[1]*frame_per_second)
		
		f0[start_idx:end_idx+1] = np.interp(np.arange(start_idx, end_idx+1), xp = [start_idx, end_idx], fp = [f0[start_idx], f0[end_idx]])
		
		'''
		for idx in range(start_idx + 1, end_idx):
			# linear interpolate
			f0[idx] = f0[start_idx] + ((f0[end_idx] - f0[start_idx])/(end_idx - start_idx))*(idx - start_idx)
		'''
	
	return f0





# TODO: Danakil Tiken conversion explodes at 35.1-35.18 ! 35.127 center, due to pitch_weight = 1, but lower it hurts stability of b_to_s
def knn_with_concat_cost(target_feature_indices, src_elements, tgt_elements, shifted_src_f0 = None, tgt_f0 = None, concat_weight = 0.2):
	
	import torch
	assert len(target_feature_indices) == len(src_elements)
	
	topk = target_feature_indices.shape[1]
	new_target_feature_indices = [target_feature_indices[0]]
	
	
	
	if shifted_src_f0 is not None:
		assert tgt_f0 is not None
		shifted_src_f0 = shifted_src_f0.to(src_elements)
		tgt_f0 = tgt_f0.to(tgt_elements)
		
		shifted_src_f0 = torch.log2(shifted_src_f0 + 1e-5)
		tgt_f0 = torch.log2(tgt_f0 + 1e-5)
	
		
	
	for i in range(1, len(src_elements)):
		
	
		# prevent exceeding tgt_elements' length
		extra_candidate_indices = new_target_feature_indices[-1] + 1
		extra_candidate_indices[extra_candidate_indices >= len(tgt_elements)] = len(tgt_elements) - 1

		
	
		
		all_candidate_indices = torch.cat([target_feature_indices[i], extra_candidate_indices])
		
		all_candidates = tgt_elements[all_candidate_indices]
		
		
		matching_cost = fast_cosine_dist(src_elements[i][None], all_candidates)
		concat_cost = fast_cosine_dist(tgt_elements[new_target_feature_indices[-1]], all_candidates)
		
		src_concat_baseline = fast_cosine_dist(src_elements[i-1][None], src_elements[i][None])[0][0]*2
		
		# prevent mix between 0 and pitched ones as we use the selected ones' harmonics for sinusoid
		if shifted_src_f0 is not None:
			all_candidate_pitches = tgt_f0[all_candidate_indices]
			
			# normalized by 1 interval
			# !!! log2 is assumed, consistent with what is in sort_by_f0_compatibility
			matching_pitch_cost = torch.abs(all_candidate_pitches[None] - shifted_src_f0[i])
		
			# print(matching_cost)
			# print("???", 0.05*matching_pitch_cost)
			# import sys
			# sys.exit()
			# matching_cost +
			
			
			if src_concat_baseline < 0.08:
				concat_cost[concat_cost < 5*src_concat_baseline] = 0
				# concat_weight = 0.2
				
				
			else:	
				
				concat_weight = 0
			
			
		
		
			total_cost = concat_weight*torch.median(concat_cost, dim = 0, keepdim = True).values + matching_cost + matching_pitch_cost
			
		else:
			concat_cost[concat_cost > src_concat_baseline] = 1.5*concat_cost[concat_cost > src_concat_baseline] - src_concat_baseline
			# concat_weight = 0.2
			total_cost = concat_weight*torch.median(concat_cost, dim = 0, keepdim = True).values + matching_cost
		
		
		
		
		topk_sort = total_cost.topk(k=topk, dim=-1, largest=False)
		
		
		final_indices = all_candidate_indices[topk_sort.indices[0]]
		new_target_feature_indices.append(final_indices)
		
		'''
		if i == int(34.58*50):
			print(new_target_feature_indices[-1])
			print(all_candidate_indices)
			# print(concat_cost)
			print("concat cost", torch.median(concat_cost, dim = 0, keepdim = True).values)
			print("cosine cost", matching_cost)
			print("weight", concat_weight)
			print("total cost", total_cost)
			print("src concat", fast_cosine_dist(src_elements[i-1][None], src_elements[i][None]))
			print("final", final_indices)
			
			# import sys
			# sys.exit()
		'''
		
	return torch.stack(new_target_feature_indices)
	


