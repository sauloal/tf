http://deeplearning.net/tutorial/lstm.html

wget http://deeplearning.net/tutorial/code/lstm.py
wget http://deeplearning.net/tutorial/code/imdb.py
wget https://raw.githubusercontent.com/kyunghyuncho/DeepLearningTutorials/master/code/imdb_preprocess.py
wget http://www.iro.umontreal.ca/~lisa/deep/data/imdb.dict.pkl.gz
wget http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl

conda install pydot-ng
conda install graphviz

THEANO_FLAGS="floatX=float32" python lstm.py


