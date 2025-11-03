# kNN-SVC: Robust Zero-Shot Singing Voice Conversion with Additive Synthesis and Concatenation Smoothness Optimization (kNN-SVC)

This repo provides inference for kNN-SVC. The project is managed with Poetry for reproducible, isolated runs.

- Prereqs: Python 3.11, Poetry
- Install deps: `poetry install`
- Checkpoints can be found under the Releases tab, place them in a folder and specify it as a command line argument (or modify it in the notebook)
- Run conversions using any of the three pathways below. Feel free to report bugs/confusion via Issues.

All examples assume 16kHz, mono audio inputs.

## 1) Single file ➜ single file

Runs the main entrypoint and saves output next to the source file as:
<src_basename>_to_<tgt_basename>_knn_<ckpt_type>_<post_opt>.wav

```bash
poetry run python ddsp_inference.py /path/to/src.wav /path/to/style.wav \
    --ckpt_dir /path/to/ckpt_dir \
    --ckpt_type mix \
    --post_opt post_opt_0.2 \
    --topk 4 \
    --device cuda \
    --prioritize_f0 true \
    --tgt_loudness_db -16
```

Notes:
- `--ckpt_type` options include: mix, mix_harm_no_amp_*, mix_no_harm_no_amp_*, wavlm_only, wavlm_only_original, where harm indicates the additive synthesis conditioning
- `--post_opt` smoothness optimization, can be `no_post_opt` or `post_opt_0.2`

## 2) Dataset ➜ dataset

Both `src` and `tgt` should be dataset roots that contain speaker subfolders.
Converted audio will be written under the parent directory of the target dataset folder, in a directory automatically created like:
`<parent_of_tgt>/{src_name}_to_{tgt_name}_{ckpt_type}_post_opt_{post_opt}/`

```bash
poetry run python ddsp_inference.py /path/to/src_dataset_root /path/to/tgt_dataset_root \
    --ckpt_dir /path/to/ckpt_dir \
    --ckpt_type mix \
    --post_opt post_opt_0.2 \
    --required_subset_file /path/to/split.csv
```

Notes:
- `--required_subset_file` can filter which files are processed (CSV format expected by the code)
- `--dur_limit` restricts the target pool to the first N minutes (set to a number or leave empty for all)


## 3) Notebook demo

Open `knnsvc_demo.ipynb` for an interactive, quick demo that uses the same `ddsp_inference.py` entrypoint under the hood.

Steps:
- Ensure you have 16kHz, mono WAVs for the source (content) and target (style).
- In the first cell, set `src_wav_path` and `ref_wav_path` and optionally tweak `ckpt_type`, `post_opt`, and `topk`.
- Launch the notebook using Poetry so it can see the project dependencies:

    ```bash
    poetry run jupyter notebook knnsvc_demo.ipynb
    ```

- Run the next cell to perform the conversion. The result will be saved next to the source file as:
    `<src_basename>_to_<tgt_basename>_knn_<ckpt_type>_<post_opt>.wav`
- Subsequent cells will load and play the result inside the notebook.

Tip: `ckpt_type` options include `mix`, `mix_harm_no_amp_*`, `mix_no_harm_no_amp_*`, `wavlm_only`, `wavlm_only_original`. `post_opt` can be `no_post_opt` or `post_opt_0.2`.


## Status and planned cleanup

We plan to standardize the `ckpt_type` naming to reduce confusion, and turn the notebook into a Colab demo, but it may depend on how this research further develops. The current options listed above will continue to work for now.


**Links**:

- Arxiv paper: [https://arxiv.org/abs/2504.05686](https://arxiv.org/abs/2504.05686)
- Demo page with samples: [http://knnsvc.com/](http://knnsvc.com/)
![kNN-SVC method](./knn-svc.png)


**Authors**:

- [Keren Shao](https://scholar.google.com/citations?user=jcQHdRgAAAAJ)
- [Ke Chen](https://www.knutchen.com/)
- [Matthew Baas](https://rf5.github.io/)
- [Shlomo Dubnov](http://dub.ucsd.edu/)


## Citation

```bibtex
@inproceedings{shao2025knn,
  title={kNN-SVC: Robust Zero-Shot Singing Voice Conversion with Additive Synthesis and Concatenation Smoothness Optimization},
  author={Shao, Keren and Chen, Ke and Baas, Matthew and Dubnov, Shlomo},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```




