# dpss-exp3-VC-PPG
Voice Conversion Experiments for THUHCSI Course : &lt;Digital Processing of Speech Signals>

## Set up environment

1. Install sox from http://sox.sourceforge.net/

2. Install sptk from http://sp-tk.sourceforge.net/

3. Set up conda environment through:
```bash
conda env create -f environment.yml
conda activate ppg-vc-env

```

## Data Preparation
1. Download CMU_ARCTIC corpus from http://festvox.org/cmu_arctic/cmu_arctic/packed/.


We don't use the Indian accent dataset, i.e., 'cmu_us_ksp_arctic' in our experiments since our ppgs-extractor is based on standard English.
For the speaker 'cmu_us_awb_arctic', choose the 0.95-release rather than 0.9-release.

Thus we have 6 speakers' dataset to use: awb, bdl, clb, jmk, rms, slt.

Extract the dataset, and organize your data directories as follows:
```bash
cmu_arctic/
├── cmu_us_awb_arctic
├── cmu_us_bdl_arctic
├── cmu_us_clb_arctic
├── cmu_us_jmk_arctic
├── cmu_us_rms_arctic
└── cmu_us_slt_arctic
```

## Any-to-One Voice Conversion Model

### Feature Extraction

```bash
# in any-to-one VC task, we use 'slt' as the target speaker.
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python preprocess.py --data_dir /path/to/cmu_arctic/cmu_us_slt_arctic --save_dir /path/to/save/cmu-arctic-slt/
```

Your extracted features will be organized as follows:
```bash
cmu-arctic-slt/
├── dev_meta.csv
├── f0s
│   ├── slt_a0001.npy
│   ├── ...
├── linears
│   ├── slt_a0001.npy
│   ├── ...
├── mels
│   ├── slt_a0001.npy
│   ├── ...
├── ppgs
│   ├── slt_a0001.npy
│   ├── ...
├── test_meta.csv
└── train_meta.csv
```

### Train

with GPU (one typical GPU is enough):
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python train_to_one.py --model_dir ./model_dir --test_dir ./test_dir --data_dir /path/to/save/cmu-arctic-slt/
```

without GPU:

```bash
python train_to_one.py --model_dir ./model_dir --test_dir ./test_dir --data_dir /path/to/save/cmu-arctic-slt/
```
### Inference

```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference_to_one.py --src_wav /path/to/source/wav --ckpt ./model_dir/ppg-vc-to-one-49.pt --save_dir ./test_dir/
```


## Any-to-Many Voice Conversion Model

### Feature Extraction

```bash
# in any-to-many VC task, we use all the above 6 speakers as the target speaker set.
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python preprocess.py --data_dir /path/to/cmu_arctic --save_dir /path/to/save/cmu_arctic-all/
```

Your extracted features will be organized as follows:
```bash
cmu-arctic-all/
├── dev_meta.csv
├── f0s
│   ├── awb_a0001.npy
│   ├── ...
├── linears
│   ├── awb_a0001.npy
│   ├── ...
├── mels
│   ├── awb_a0001.npy
│   ├── ...
├── ppgs
│   ├── awb_a0001.npy
│   ├── ...
├── test_meta.csv
└── train_meta.csv
```

### Train

with GPU (one typical GPU is enough):
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python train_to_many.py --model_dir ./model_dir --test_dir ./test_dir --data_dir /path/to/save/cmu_arctic-all/
```

without GPU:

```bash
python train_to_many.py --model_dir ./model_dir --test_dir ./test_dir --data_dir /path/to/save/cmu_arctic-all/
```
### Inference

```bash
# here for inference, we use 'slt' as the target speaker. you can change the tgt_spk argument to any of the above 6 speakers. 
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference_to_many.py --src_wav /path/to/source/wav --tgt_spk slt --ckpt ./model_dir/ppg-vc-to-many-49.pt --save_dir ./test_dir/
```

## Assignment requirements
This project is a vanilla voice conversion system based on PPG. 

When you encounter problems while finishing your project, search the [issues](https://github.com/thuhcsi/dpss-exp3-VC-PPG/issues) first to see if there are similar problems. If there are no similar problems, you can create new issues and state you problems clearly.
