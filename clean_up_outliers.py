import glob
import os
import shutil

files = glob.glob("./data/ava_activespeaker_samples/train/*")
count = 0
for each_file in files:
    total_frames = len(os.listdir(each_file))
    if(total_frames != 15):
        print("Problem in {}".format(each_file))
        shutil.rmtree(each_file)
        count += 1
print("Total problems: {}".format(count))