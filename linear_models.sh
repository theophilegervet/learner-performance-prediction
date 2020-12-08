echo "writing all files"

for filename in assistments09 assistments12 assistments15 assistments17 bridge_algebra06 algebra05 spanish statics
do
    python encode.py --dataset $filename -i -s -ic -sc -tc -w -a
    echo $filename
    echo "done"
    echo "---------"
done