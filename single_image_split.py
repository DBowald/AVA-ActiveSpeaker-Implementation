import numpy as np
import cv2
import pandas as pd
import glob
import json
import random

# Open a sample videos available in sample-videos
#vcap = cv2.VideoCapture('https://s3.amazonaws.com/ava-dataset/trainval/TzaVHtLXOzY.mkv')
#data_frame = pd.read_csv('./data/ava_activespeaker_train_v1.0/TzaVHtLXOzY-activespeaker.csv')
#vcap = cv2.VideoCapture('https://s3.amazonaws.com/ava-dataset/trainval/P60OxWahxBQ.mkv')
#data_frame = pd.read_csv('./data/ava_activespeaker_test_v1.0/P60OxWahxBQ-activespeaker.csv')
#if not vcap.isOpened():
#    print "File Cannot be Opened"

lookup = dict()
for line in open('filenames.txt'):
    key, value = line.strip().split('.')
    lookup[key] = value

for partition in ["test", "train"]:
    all_csvs = glob.glob("./data/ava_activespeaker_{}_v1.0/*.csv".format(partition))
    for each_csv in all_csvs:
        try:
            youtube_id = each_csv[35:-18]
            video_url = 'https://s3.amazonaws.com/ava-dataset/trainval/{}.{}'.format(youtube_id, lookup[youtube_id])
            vcap = cv2.VideoCapture(video_url)
            data_frame = pd.read_csv(each_csv)

            frame_width = int(round(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            frame_height = int(round(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = vcap.get(cv2.CAP_PROP_FPS)

            print("FRAME WIDTH ", frame_width)
            print("FRAME HGITH ", frame_height)
            print("FPS", fps)

            prev_frame = 0

            for i, each_row in data_frame.iterrows():
                # Capture frame-by-frame
                time = each_row[1]
                current_frame = int(round(time*fps))
                top_left = (int(round(each_row[2] * frame_width)), int(round(each_row[3] * frame_height)))
                bot_right = (int(round(each_row[4] * frame_width)), int(round(each_row[5] * frame_height)))
                if each_row[6] == 'SPEAKING_AUDIBLE' or each_row[6] == 'SPEAKING_NOT_AUDIBLE':
                    label = "speaking"
                elif each_row[6] == 'NOT_SPEAKING':
                    label = "not_speaking"
                else:
                    print(each_row[6])
                    continue
                #print(label)
                if(current_frame - prev_frame > 1):
                    vcap.set(1, current_frame)
                ret, frame = vcap.read()
                if(random.random() > 0.033333):
                    continue
                roi = frame[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
                prev_frame = current_frame
                #vcap.set(1, 2000)
                #print cap.isOpened(), ret
                save_path = "./data/single_image_activespeaker_{}/{}/{}_{}.png".format(
                    partition,label,youtube_id,current_frame)
                cv2.imwrite(save_path, roi)
            # When everything done, release the capture
            vcap.release()
        except Exception:
            continue