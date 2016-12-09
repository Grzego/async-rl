### Variation of Asynchronous RL in Keras (Theano backend) + OpenAI gym [1-step Q-learning]
This is a simple variation of [asynchronous reinforcement learning](http://arxiv.org/pdf/1602.01783v1.pdf) written in Python with Keras (Theano backend).
Instead of many threads training at the same time there are many processes generating experience for a single agent to learn from.

### Explanation
There are many processes (tested with 4, it probably works better with more) which are creating experience and sending it to the shared queue. Queue is limited in length (tested with 256) to stop individual processes from excessively generating experience with old weights. Learning process draws from queue samples in batches (tested with 64) and learns on them.

### Usage
Requirements:
* [Python 3.4/Python 3.5](https://www.python.org/downloads/)
* [Keras](http://keras.io/)
* [Theano](http://deeplearning.net/software/theano/) ([Tensorflow](https://www.tensorflow.org/) would probably work too)
* [OpenAI (atari-py)](https://gym.openai.com/)

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

![](https://raw.githubusercontent.com/Grzego/multiprocess-rl/master/q-learning-1-step/resources/after-6h-training.gif?token=AFhQOQQq2JlswCS_p1XjU6WrKn3pQ4dvks5XbsV9wA%3D%3D)
![](https://raw.githubusercontent.com/Grzego/multiprocess-rl/master/q-learning-1-step/resources/after-12h-training.gif?token=AFhQOXkCZbPO9SrOXXu5_3_-P0ftrfSsks5XbsWiwA%3D%3D)
![](https://raw.githubusercontent.com/Grzego/multiprocess-rl/master/q-learning-1-step/resources/after-18h-training.gif?token=AFhQOR-kTbupToKnNRenZCWiBEtZBmvhks5XbsWjwA%3D%3D)

### Previous mistakes
I overlooked that in paper they don't use old weights to stabilize learning. To fix this I removed network holding old weights and set swapping frequency to 100 and batch size to 20. With this setting after 5 batches every process gets new weights. Now agent should learn much quicker compared to old implementation.

#### Feedback
Because I'm newbie in Reinforcement Learning and Deep Learning, feedback is very welcome :)

### Note
* You can find weights for model in those stages in `sample-weights` folder (trained with old version of this code).
* Weights were learned in Theano, so loading them in Tensorflow may be a little problematic due to Convolutional Layers.
* If training halts after few seconds, don't worry, its probably because Keras lazily compiles Theano function, it should resume quickly.
* In different folders you can find other techniques like [n-step Q-learning](https://github.com/Grzego/async-rl/tree/master/q-learning-n-step) or [A3C](https://github.com/Grzego/async-rl/tree/master/a3c).
* Each process sets its own compilation directory for Theano so compilation can take very long time at the beginning (can be disabled with `--th_comp_fix=False`)

### Useful resources
* [Asyncronous RL in Tensorflow + Keras + OpenAI's Gym](https://github.com/coreylynch/async-rl)
* [Replicating "Asynchronous Methods for Deep Reinforcement Learning"](https://github.com/muupan/async-rl)
* [David Silver's "Deep Reinforcement Learning" lecture](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/pdf/1602.01783v1.pdf)
* [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/pdf/1312.5602v1.pdf)


#### Important changes
* Switched from Python 2.7 to Python 3.4
* Switched to multiprocessing.Pool for better process management (to avoid zombie processes)