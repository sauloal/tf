python run.py 2>&1 | tee log.log &

tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
