# One Model Any Conjunctive Query

This repository contains codebase for the paper "One Model, Any Conjunctive Query: Graph Neural Networks for Answering Complex Queries over Incomplete Knowledge Graphs". The code is based on [ANYCSP](https://github.com/toenshoff/ANYCSP).

## Setting up the environment

Create a new virtual environment and install dependencies with pip/conda (change the pytorch version if needed):
```
conda create --name anycq python=3.10
conda activate anycq
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Download models and training data

To download the training data, in the main folder run:
```
gdown https://drive.google.com/uc?id=1Ib7HE4oBBCDeH9yeu3dfwVk3YIHZ4kSO
unzip training_data.zip
```
Alternatively, the data can be accessed directly on [Google Drive](https://drive.google.com/file/d/1Ib7HE4oBBCDeH9yeu3dfwVk3YIHZ4kSO/view). Put each `train_qaa.json` file in the correct subfolder of `data`.

For downloading models' checkpoints, execute:
```
gdown https://drive.google.com/uc?id=1XhtwwmcbWhatS7K6VIzvNr5EcJIJQqZq
unzip models.zip
```
or access them manually on [Google Drive](https://drive.google.com/file/d/1XhtwwmcbWhatS7K6VIzvNr5EcJIJQqZq/view?usp=sharing).


## Training AnyCQ
Training scripts are available in the `scripts` folder:
```
# NELL
bash scripts/train_anycq_nell.sh
# FB15k-237
bash scripts/train_anycq_fb15k_237.sh
```
The respective configuration files are `configs/model/model_NELL.json` and `configs/model/model_FB15k-237.json`.

## Testing AnyCQ
We provide three validation scripts: for QAR, small QAC and large QAC splits.

### QAR
AnyCQ can be tested on the QAR task by running the `test_anycq_qar.py` script:
```
python test_anycq_qar.py --model_dir "models/FB15k-237/anycq/" --model_name "_checkpoint_350000" --config_file "configs/model/model_FB15k-237_qar.json" --exp_name "Test_FB15k-237_3hub_" --n_pivots 3
```
where the arguments are responsible for:
- `model_dir`: path to the pre-trained model checkpoint
- `model_name`: name of the checkpoint
- `config_file`: the config file jointly representing the model and dataset
- `exp_name`: name of the experiments to be logged in the `logs` folder
- `n_pivots`: 3/4/5, denoting different splits

Changing the config file allows to include inference with equipped perfect link predictor. By changing the model name and directory, one can check the transferability of different models to the dataset specified in config.

Additionally, for enabling parallel computation one can also add the following flags:
- `start`: the id of the first instance to be processed (inclusive)
- `end`: the id of the first instance to be processed (exclusive)

For example, the following command can be called to process the first 100 queries in the 4-hub split:
```
python test_anycq_qar.py --model_dir "models/FB15k-237/anycq/" --model_name "_checkpoint_350000" --config_file "configs/model/model_FB15k-237_qar.json" --exp_name "Test_FB15k-237_4hub_" --n_pivots 4 --start 0 --end 100
``` 

### QAC
We split the QAC evaluation into two parts, considering small and large splits separately. To run testing on simple queries:
```
# Small QAC
python test_anycq_qac_small.py --model_dir "models/FB15k-237/anycq/" --model_name "_checkpoint_350000" --config_file "configs/model/model_FB15k-237_qac_small.json" --exp_name "Test_small_QAC_FB15k-237_"
```

Running QAC testing on large splits requires one additional parameter `--gen_type`, taking values in `[3hub, 4hub, 5hub]`, depending on the evaluated split.

```
# Large QAC
python test_anycq_qac.py --model_dir "models/NELL/anycq/" --model_name "_checkpoint_200000" --config_file "configs/model/model_NELL_qac.json" --exp_name "QAC_Test_NELL_5hub_" --gen_type 5hub --end 75
```

The purpose of each flag is identical to the QAR setting. Once again, defining `--start/--end` flags, one can modify the range of processed queries.

## Testing the SQL Engine
To test the performance of the SQL DuckDB engine for the QAC task, simply run:
```
python test_sql_qac.py --dataset NELL
python test_sql_qac.py --dataset FB15k-237
```

Similarly, to the performance on the QAR benchmarks, execute:
```
# For dataset FB15k-237-QAR and the split 3hub:
python test_sql_qar.py --dataset FB15k-237 --gen_type 3hub
```
