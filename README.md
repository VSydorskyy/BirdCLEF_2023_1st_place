# Bird Clef 2023 1st place solution
Codebase for Bird Clef 2023 1st place solution

# Pre-requirements 

- Ubuntu 22.04.2 LTS
- CUDA Version: 12.0
- CUDA Driver Version: 525.105.17
- conda 22.9.0
- GPU A100 40Gb (but any other GPU, compatible with CUDA Version and with VRAM >= 10Gb, should work)
- Hard Disk: 1.5 Tb
- RAM: 500 Gb

# Main Pipeline 
```bash
# Export credentials
 export KAGGLE_USERNAME={KAGGLE_USERNAME}
 export KAGGLE_KEY={KAGGLE_KEY}
# Create environment
conda env create -f requirements/environment.yml
conda activate bird_clef_2023_1st_place
pip install -e .
# Run Pipeline
bash rock_that_bird.sh "{GPU_TO_USE}" # By default: bash rock_that_bird.sh "0"
```

Find final model in `logdirs/convnext_small_fb_in22k_ft_in1k_384__convnextv2_tiny_fcmae_ft_in22k_in1k_384__eca_nfnet_l0_noval_v32_075Clipwise025TimeMax_GausMean/onnx_ensem/model_simpl.onnx`
