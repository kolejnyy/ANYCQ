

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
python test_anycq_qar.py --model_dir "models/FB15k-237/anycq/" --model_name "_checkpoint_350000" --config_file "configs/model/model_FB15k-237_qar.json" --exp_name "Test_FB15k-237_4hub_" --n_pivots 3
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
```
# Small QAC


# Large QAC
python test_anycq_qac.py --model_dir "models/FB15k-237-EFO1/anycq/" --model_name "_checkpoint_350000" --config_file "configs/model/model_NELL_qac.json" --exp_name "QAC_Test_NELL_5hub_0_" --gen_type 5hub --end 75
```

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