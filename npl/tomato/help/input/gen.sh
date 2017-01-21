if [[ ! -f "mnist-to-jpg.py" ]]; then
wget https://gist.githubusercontent.com/ischlag/41d15424e7989b936c1609b53edd1390/raw/5ed7aca47bcca30b3df1c3bfd0f027e6bcdb430c/mnist-to-jpg.py
fi


python mnist-to-jpg.py

python run.py | tee log.log
