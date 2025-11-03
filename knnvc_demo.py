# %%
# path to 16kHz, single-channel, source waveform
src_wav_path = '/content/src.wav'
# list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
ref_wav_paths = ['/content/ref1.wav', ]

print("hahaha")
exit

query_seq = knn_vc.get_features(src_wav_path)
matching_set = knn_vc.get_matching_set(ref_wav_paths)

# %%
import torch, torchaudio

knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')

# %%
out_wav = knn_vc.match(query_seq, matching_set, topk=4)

# %%
import IPython.display as ipd

# %%
ipd.Audio(out_wav.numpy(), rate=16000)

# %%
a = 1+1
print(a)

# %% [markdown]

# But the above does not make any sense though.

# %%
torchaudio.save('knnvc1_out.wav', out_wav[None], 16000)

# %% [markdown]
# <audio name="abstract-reader" controls preload src="https://github.com/bshall/knn-vc/releases/download/v0.1/david-attenborough.wav"></audio>


# %%
