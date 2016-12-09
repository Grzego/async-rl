### Usage

To start training simply type (I recommend running in terminal with maximum width, due to lots of output data):
```
python train.py --game=Breakout-v0 --processes=16
```

To resume training from saved model (ex. `model-1250000.h5`):
```
python train.py --game=Breakout-v0 --processes=16 --checkpoint=1250000
```

To see how it plays:
```
python play.py --model=sample-weights/model-18h.h5 --game=Breakout-v0
```

### Samples (old version)
I tested it once and it worked quite well. (Intel i7-4700MQ and NVidia GTX 765M)

Sample games after 6h, 12h and 18h of training.

![](https://raw.githubusercontent.com/Grzego/async-rl/master/q-learning-1-step/resources/after-6h-training.gif?token=AFhQOQQq2JlswCS_p1XjU6WrKn3pQ4dvks5XbsV9wA%3D%3D)
![](https://raw.githubusercontent.com/Grzego/async-rl/master/q-learning-1-step/resources/after-12h-training.gif?token=AFhQOXkCZbPO9SrOXXu5_3_-P0ftrfSsks5XbsWiwA%3D%3D)
![](https://raw.githubusercontent.com/Grzego/async-rl/master/q-learning-1-step/resources/after-18h-training.gif?token=AFhQOR-kTbupToKnNRenZCWiBEtZBmvhks5XbsWjwA%3D%3D)