# OTscore Project

This project contains:
- `train_source.py`: train source model (`source_F/B/C_<seed>.pt`)
- `train_target.py`: adapt target model from source checkpoint (`target_F/B/C_<seed>.pt`)
- `utils/ot_score_utils.py`: OT score and feature/prototype utilities

## 1) Environment

Create environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate OTscore
```

If your conda env name is different (for example `DA`), use that env consistently.
In commands below, replace `/path/to/domainnet/` with your own dataset root directory.

## 2) Data Layout

Current code assumes `--datadir` contains both:
- image folders referenced by list files
- DomainNet list files like:
  - `clipart_list.txt`
  - `real_list.txt`
  - `painting_list.txt`
  - `sketch_list.txt`

Each list file line format:

```text
relative/or/absolute/image_path label_id
```

## 3) Train Source

Run one source-target task by indices (`--s`, `--t`) using DomainNet:

```bash
python train_source.py --dset domainnet --s 0 --t 1 --datadir "/path/to/domainnet/" --output output
```

Only train source (do not run `test_target`) by default.  
To run source-side test after training:

```bash
python train_source.py --dset domainnet --s 0 --t 1 --datadir "/path/to/domainnet/" --output output --test_target
```

Source checkpoints are saved to:

```text
output/<dset>/<SourceInitial>/
  source_F_<seed>.pt
  source_B_<seed>.pt
  source_C_<seed>.pt
```

## 4) Train Target Adaptation

DomainNet example:

```bash
python train_target.py --dset domainnet --s 0 --t 1 --datadir "/path/to/domainnet/" --output_src output --output output --seed 2020
```

Target script loads source checkpoints from:

```text
<output_src>/<dset>/<SourceInitial>/source_*.pt
```

Target checkpoints are saved to:

```text
<output>/<dset>/<SourceInitial><TargetInitial>/
  target_F_<seed>.pt
  target_B_<seed>.pt
  target_C_<seed>.pt
```
