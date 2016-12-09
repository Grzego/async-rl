#### Usage

To start training simply type:
```
python train.py --game=Breakout-v0 --processes=16
```

To resume training from saved model (ex. `model-Breakout-v0-1250000.h5`):
```
python train.py --game=Breakout-v0 --processes=16 --checkpoint=1250000
```

To see how it plays:
```
python play.py --model=model-file.h5 --game=Breakout-v0
```

### Results

This method works really well. Graph below shows average score of 10 games played every 1kk frames. Learning took about 24h. I was able to process ~57k frames every minute. Final weights can be found in `sample-weights` folder.

![](https://github.com/Grzego/async-rl/blob/master/a3c/resources/average-scores.png?raw=true)

### Sample game

![](https://github.com/Grzego/async-rl/blob/master/a3c/resources/sample-game.gif?raw=true)