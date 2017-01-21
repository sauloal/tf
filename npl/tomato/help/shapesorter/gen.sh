#create folders
mkdir -p data/train/squares
mkdir -p data/train/triangles
mkdir -p data/validate/squares
mkdir -p data/validate/triangles

#gen images
python create.py

#copy training set
mv data/train/squares/data3*    data/validate/squares/.
mv data/train/triangles/data3*  data/validate/triangles/.

cd data

echo -e "squares\ntriangles" > mylabels.txt

if [[ ! -f "build_image_data.py" ]]; then
	wget https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/build_image_data.py
fi

python build_image_data.py \
	--train_directory=./train \
	--output_directory=./  \
	--validation_directory=./validate \
	--labels_file=mylabels.txt \
	--train_shards=1 \
	--validation_shards=1 \
	--num_threads=1

cd ..

if [[ ! -f "shapesorter.py" ]]; then
	wget https://agray3.github.io/files/shapesorter.py
fi

python shapesorter.py | tee log.log
