function get {
	echo GETTING $1
	bn=`basename $1`
	if [[ ! -f "$bn" ]]; then
		wget $1
	else
		echo "$bn exists"
	fi
}

get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10.py
get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10_eval.py
get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10_input.py
get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10_input_test.py
get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
get https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/cifar10/cifar10_train.py

python cifar10_train.py

