# Bird Clef 2023 1st place solution
Codebase for Bird Clef 2023

```bash
# Create environment
conda env create -f requirements/environment.yml
conda activate bird_clef_2023_1st_place
pip install -e .
# Export credentials
 export KAGGLE_USERNAME={KAGGLE_USERNAME}
 export KAGGLE_KEY={KAGGLE_KEY}
# Download data and prepare
cd data
kaggle competitions download -c birdclef-2023
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part1
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part2
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part3
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part4
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part5
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part6
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part7
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part8

unzip birdclef-2023.zip -d birdclef_2023
rm birdclef-2023.zip

unzip birdclef-2023-data-part1.zip -d birdclef_2023_data_part1
rm birdclef-2023-data-part1.zip

unzip birdclef-2023-data-part2.zip -d birdclef_2023_data_part2
rm birdclef-2023-data-part2.zip

unzip birdclef-2023-data-part3.zip -d birdclef_2023_data_part3
rm birdclef-2023-data-part3.zip

unzip birdclef-2023-data-part4.zip -d birdclef_2023_data_part4
rm birdclef-2023-data-part4.zip

unzip birdclef-2023-data-part5.zip -d birdclef_2023_data_part5
rm birdclef-2023-data-part5.zip

unzip birdclef-2023-data-part6.zip -d birdclef_2023_data_part6
rm birdclef-2023-data-part6.zip

unzip birdclef-2023-data-part7.zip -d birdclef_2023_data_part7
rm birdclef-2023-data-part7.zip

unzip birdclef-2023-data-part8.zip -d birdclef_2023_data_part8
rm birdclef-2023-data-part8.zip

mv birdclef_2023_data_part2/soundscapes_nocall.zip.part-* ./
mv birdclef_2023_data_part3/soundscapes_nocall.zip.part-* ./
cat soundscapes_nocall.zip.part-* > soundscapes_nocall.zip
rm soundscapes_nocall.zip.part-*
unzip soundscapes_nocall.zip
rm soundscapes_nocall.zip

mv birdclef_2023_data_part6/xeno_canto.zip.part-* ./
mv birdclef_2023_data_part7/xeno_canto.zip.part-* ./
cat xeno_canto.zip.part-* > xeno_canto.zip
rm xeno_canto.zip.part-*
unzip xeno_canto.zip
rm xeno_canto.zip

mv birdclef_2023_data_part1/audio/audio audio
rm birdclef_2023_data_part1/audio -r
mv birdclef_2023_data_part1/* ./

mv birdclef_2023_data_part2/esc50/esc50 esc50

mv birdclef_2023_data_part3/birdclef_2020/birdclef_2020 birdclef_2020

mv birdclef_2023_data_part4/birdclef_2020_xc_a_m/birdclef_2020_xc_a_m birdclef_2020_xc_a_m
mv birdclef_2023_data_part4/birdclef_2020_xc_n_z/birdclef_2020_xc_n_z birdclef_2020_xc_n_z

mv birdclef_2023_data_part5/birdclef_2021 birdclef_2021

mv birdclef_2023_data_part8/birdclef_2022 birdclef_2022

rm birdclef_2023_data_part* -r
```

## Setup Env

1. `conda env create -f requirements/environment.yml`
2. `conda activate bird_clef_2023`
4. `pip install -e .`

## Data Preparation

1. Download 2022 train_data - `kaggle competitions download -c birdclef-2022` and unzip
2. Download Selim Maupar FIX `https://www.kaggle.com/datasets/realsleim/maupar-fix` and unzip
3. Inplace `maupar` sample with splitted one in `train_audio`
3. Download previous year training data `kaggle competitions download -c birdclef-2021` and unzip
4. Took all files, which are present in `data/train_metadata_processed_rocpig1fixed.csv` and put into your `train_audio`
5. Modify `filename` of  `data/train_soundscape_labels_processed.csv` with your path to 2021 soundscapes (Sorry fot this shit :( ))
6. Download addintional XC data - `https://www.kaggle.com/datasets/vladimirsydor/xc-scored-add` and unzip
7. Took all files, which are present in `data/train_metadata_nodupl_v3.csv` and put into your `train_audio`
8. Download background data - `https://www.kaggle.com/datasets/vladimirsydor/birdclef-background` and unzip
10. run `python scripts/precompute_features.py {path_to_your_train_audio} {path_to_your_train_features}`


## Start Pre-Training stage

1. Open `train_configs_best/best_effnet_pretrain.py` OR `train_configs_best/best_nfnet_pretrain.py`
2. Inplace `???_main_root` with `{path_to_your_train_features}`
3. Inplace `???_esc50_root` with `{path_to_background_data}/audio` (From Data Preparation Step 8)
4. Start training - `CUDA_VISIBLE_DEVICES={YOUR DEVICES} python scripts/main_train.py train_configs_best/best_effnet_pretrain.py` OR `CUDA_VISIBLE_DEVICES={YOUR DEVICES} python scripts/main_train.py train_configs_best/best_nfnet_pretrain.py`

## Start Fine-Tuning stage

1. Open `train_configs_best/best_effnet_finetune.py` OR `train_configs_best/best_nfnet_finetune.py`
2. Inplace `???_main_root` with `{path_to_your_train_features}`
3. Inplace `???_esc50_root` with `{path_to_background_data}/audio` (From Data Preparation Step 8)
4. Start training - `CUDA_VISIBLE_DEVICES={YOUR DEVICES} python scripts/main_train.py train_configs_best/best_effnet_finetune.py` OR `CUDA_VISIBLE_DEVICES={YOUR DEVICES} python scripts/main_train.py train_configs_best/best_nfnet_finetune.py`
