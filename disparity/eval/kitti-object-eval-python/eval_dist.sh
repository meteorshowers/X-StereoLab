#!/bin/bash
echo $1
if [ ! -n "$2" ] ; then
    class="0"
else
    class=$2
fi
echo $class

for i in $(seq 0 5 45)
do
	echo "eval $i,$(($i+5)) meters"
	python3.6 evaluate.py evaluate \
	    --label_path=/home/yilunchen/data/kitti/training/label_2/ \
	    --result_path=$1 \
	    --current_class=$class --coco=False \
	    --eval_dist=$i,$(($i+5))
done

