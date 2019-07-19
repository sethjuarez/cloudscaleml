REM python fetch.py -d data -t data -c tacos burrito -f
python prep.py --data_path data --output_path data --target_output data
REM python train.py -d train -e 3 -b 32 -l 0.0001 -o model -f train.txt