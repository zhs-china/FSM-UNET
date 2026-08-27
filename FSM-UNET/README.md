<div align="center">
<h1>FSM-UNet</h1>
<h3>Frequency-Spatial Mamba UNet for Lightweight Skin Lesion Segmentation</h3>

</div>

## Highlights
### *1. FSM-UNet achieves competitive segmentation performance with only ~0.030M parameters and ~0.032 GFLOPs.*
### *2. Grouped Shared State Mamba (GSS-Mamba) significantly reduces parameters while preserving global modeling capacity.*
### *3. Frequency-Spatial Dual Attention ASPP (FSDA-ASPP) provides complementary frequency-domain and spatial-domain enhancement at near-zero parameter cost.*
### *4. Plug-and-play module design: GSS-Mamba, FSDA-ASPP, and DLESkip can be independently toggled via configuration.*

**0. Main Environments.** </br>
The environment installation procedure can be followed by [VM-UNet](https://github.com/JCruan519/VM-UNet), or by following the steps below (python=3.8):</br>
```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Configuration.** </br>
All training and model parameters are managed in `configs/config_setting.py`. Key settings:

```python
model_config = {
    'num_classes': 1,
    'input_channels': 3,
    'c_list': [8, 16, 24, 32, 48, 64],
    # Ablation switches (toggle modules on/off)
    'use_dle_skips': True,     # DLESkip skip refinement
    'use_gssm_blocks': True,  # GSS-Mamba blocks in encoder/decoder
    'use_fsda_aspp': True,    # FSDA-ASPP at bottleneck
    'deep_supervision': False, # Deep supervision training
    'gssm_r': 4,               # GSS-Mamba compression ratio
    'gssm_g': 2,               # GSS-Mamba group count
}
```

To switch datasets, change the `datasets` field and uncomment the corresponding `data` loader in `loader.py`.

**2. Train FSM-UNet.** </br>

Before training, set the `ASPP_VARIANT` at the top of `train.py`:
```python
ASPP_VARIANT = "D"  # A: Baseline, B: Freq-Only, C: Spatial-Only, D: Full (default)
```

Then run:
```
python train.py
```
- After training, you could obtain the outputs in `./results/` </br>
- The FLOPs and parameter count are automatically printed at the start of training.

**3. Test FSM-UNet.** </br>
First, in the `test.py` file, set the `resume_model` path to your checkpoint:
```python
resume_model = os.path.join('path/to/your/best-epochXXX-lossX.XXXX.pth')
```
Ensure `ASPP_VARIANT` in `test.py` matches the value used during training. Then run:
```
python test.py
```
- After testing, you could obtain the outputs in `./results/` </br>

**4. Inference on Custom Folders.** </br>
For inference on a custom image folder with contour overlay visualization:

1. Set the checkpoint path and dataset folder in `configs/config_setting.py`:
```python
test_weights = '/absolute/path/to/your/checkpoint.pth'
custom_test_dir = '/absolute/path/to/your/test_folder'
```
2. Ensure the folder structure:
```
test_folder/
  image/    # original images (RGB, any common format)
    001.png
    002.png
  label/    # ground truth masks (single-channel binary, same filenames)
    001.png
    002.png
```
3. Set `ASPP_VARIANT` in `infer_folder.py` to match training, then run:
```
python infer_folder.py
```
- Output: Contour overlay images saved to `{work_dir}/contour_overlay/` with red contours for ground truth and blue contours for predictions.
- Metrics: mIoU, Dice, Accuracy, Sensitivity, Specificity are computed and logged.

## Citation
If you find this repository helpful, please consider citing: </br>

## Acknowledgement
Thanks to [UltraLight VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet), [VM-UNet](https://github.com/JCruan519/VM-UNet), [Mamba](https://github.com/state-spaces/mamba) and [LightM-UNet](https://github.com/MrBlankness/LightM-UNet) for their outstanding work.
