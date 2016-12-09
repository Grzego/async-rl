### Usage

To start training simply type:
```
python train.py --game=Breakout-v0 --processes=16 --n_step=5
```

To resume training from saved model (ex. `model-1250000.h5`):
```
python train.py --game=Breakout-v0 --processes=16 --checkpoint=1250000
```

To see how it plays:
```
python play.py --model=model-file.h5 --game=Breakout-v0
```