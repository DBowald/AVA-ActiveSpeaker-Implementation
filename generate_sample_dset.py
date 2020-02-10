import cv2
import random
import numpy as np
import os
import shutil
from pathlib import Path
from argparse import ArgumentParser
import glob
import wget
import pandas as pd

lookup = dict()
for line in open('filenames.txt'):
    key, value = line.strip().split('.')
    lookup[key] = value

def check_video_downloaded(vid_path, youtube_id):
    path = os.path.join(vid_path,youtube_id + "." + lookup[youtube_id])
    if not os.path.exists(path):
        video_url = 'https://s3.amazonaws.com/ava-dataset/trainval/{}.{}'.format(youtube_id, lookup[youtube_id])
        print("Downloading {} to {}".format(video_url, path))
        wget.download(video_url, path)


def create_samples(csv_path, vid_path="./videos", temporal_window=2, overlap=1, min_frames=15, sample_freq=20):
    samples = []

    all_csvs = glob.glob(os.path.join(csv_path, "*.csv"))
    for each_csv in all_csvs:
        youtube_id = each_csv[-29:-18].strip()
        check_video_downloaded(vid_path, youtube_id)
        #video_url = 'https://s3.amazonaws.com/ava-dataset/trainval/{}.{}'.format(youtube_id, lookup[youtube_id])
        video_filepath = os.path.join(vid_path, youtube_id + "." + lookup[youtube_id])
        vcap = cv2.VideoCapture(video_filepath)
        data_frame = pd.read_csv(each_csv)

        frame_width = int(round(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_height = int(round(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = vcap.get(cv2.CAP_PROP_FPS)

        current_face_track = None
        prev_frame = None
        prev_frame_time = None
        start_time = 0
        curr_time = start_time
        end_time = curr_time + temporal_window + overlap
        next_sample = start_time
        frames_buffer = []
        overlap_buffer = []
        end_sample = False

        print("Processing {}".format(youtube_id))


        for i, each_row in data_frame.iterrows():
            face_track_id = each_row[7]
            if current_face_track is None:
                current_face_track = face_track_id
            if current_face_track != face_track_id:
                current_face_track = face_track_id
                end_sample = True

            curr_time = each_row[1]
            curr_frame_num = int(round(curr_time * fps))
            top_left = (int(round(each_row[2] * frame_width)), int(round(each_row[3] * frame_height)))
            bot_right = (int(round(each_row[4] * frame_width)), int(round(each_row[5] * frame_height)))
            #label = 0 if each_row[6] == "NOT_SPEAKING" else 1
            label = each_row[6]
            curr_frame = [youtube_id, curr_frame_num, top_left, bot_right, label, curr_time, i]

            if curr_time >= end_time:
                end_sample = True
            if prev_frame_time is not None:
                if curr_time < prev_frame_time:
                    end_sample = True
                    overlap_buffer.clear()
                    prev_frame = None

            if end_sample:
                if prev_frame is not None:
                    frames_buffer.append(prev_frame)
                    if overlap > 0:
                        overlap_buffer.append(prev_frame)

                start_time = curr_time - (curr_time % temporal_window)
                end_time = start_time + temporal_window + overlap
                next_sample = curr_time - (curr_time % (1/sample_freq)) + 1/sample_freq

                if(len(frames_buffer) < min_frames):
                    frames_buffer.clear()
                    overlap_buffer.clear()
                else:
                    samples.append(frames_buffer)
                    frames_buffer = overlap_buffer
                    overlap_buffer = []
                prev_frame = None
                prev_frame_time = None
                end_sample = False

            if prev_frame is None:
                prev_frame = curr_frame
                prev_frame_time = curr_time
                continue

            if curr_time - next_sample >= 0:
                if abs(curr_time - next_sample) < abs(prev_frame_time - next_sample):
                    winner = curr_frame
                    prev_frame = None
                    prev_frame_time = None
                else:
                    winner = prev_frame
                    prev_frame = curr_frame
                    prev_frame_time = curr_time

                frames_buffer.append(winner)
                if curr_time > end_time - overlap:
                    overlap_buffer.append(winner)
                next_sample += 1/sample_freq
            elif abs(curr_time - next_sample) >= 2/sample_freq:
                raise Exception("BWUH")

        vcap.release()
        #break
    return samples

def main(args):
    #csv_dir = "./data/ava_activespeaker_test_v1.0"
    #out_dir = "./data/ava_activespeaker_samples/test"

    # n_samples = 4000
    # frames_per_sample = 15

    if(args.overwrite_existing_dset):
        for root,dirs,files in os.walk(args.out_dir):
            for dir in dirs:
                shutil.rmtree(os.path.join(root,dir))

    csv_dir = args.csv_dir
    out_dir = args.out_dir
    n_samples = args.n_samples
    frames_per_sample = args.frames_per_sample
    samples = create_samples(csv_dir)
    vcaps = dict()
    #vcap = cv2.VideoCapture("./data/videos/053oq2xB3oU.mkv")

    total_speaking = 0
    total_not = 0
    total_other = 0
    total_samples = 0
    max_disparity = 200

    while(total_samples < n_samples):
        random_order = np.random.choice(len(samples), len(samples), replace=False)
        for each_idx in random_order:
            if(total_samples >= n_samples):
                break
            each_sample = samples[each_idx]
            num_speaking = 0
            num_not = 0

            start_idx = random.randint(0, len(each_sample)-frames_per_sample)
            youtube_id = each_sample[0][0]
            if youtube_id not in vcaps:
                cap_path = os.path.join(args.video_dir, youtube_id + "." + lookup[youtube_id])
                vcaps[youtube_id] = cv2.VideoCapture(cap_path)
            vcap = vcaps[youtube_id]

            for each_image in each_sample[start_idx:start_idx+frames_per_sample]:
                label = each_image[4]
                if(label == "SPEAKING_AUDIBLE" or label == "SPEAKING_NOT_AUDIBLE"):
                    num_speaking += 1
                elif(label == "NOT_SPEAKING"):
                    num_not += 1
                else:
                    total_other += 1
                    print(each_image[4])

            if total_speaking - total_not > max_disparity and num_speaking > num_not:
                continue
            elif total_not - total_speaking > max_disparity and num_not > num_speaking:
                continue
            else:
                dirname = youtube_id + "-" + str(total_samples)
                Path(os.path.join(out_dir, dirname)).mkdir()
                prev_frame = -1
                for each_image in each_sample[start_idx:start_idx+frames_per_sample]:
                    label = each_image[4]
                    curr_frame = each_image[1]
                    curr_time = each_image[5]
                    top_left = each_image[2]
                    bot_right = each_image[3]
                    if(curr_frame > prev_frame+1):
                        vcap.set(1, curr_frame)
                    ret, frame = vcap.read()
                    roi = frame[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
                    prev_frame = curr_frame
                    save_path = os.path.join(out_dir, dirname, "{}-{}.png".format(str(curr_time), label))
                    cv2.imwrite(save_path, roi)

                total_speaking += num_speaking
                total_not += num_not
                total_samples += 1

    print(total_speaking)
    print(total_not)
    print(total_other)
    print(total_samples)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--n_samples", type=int, default=4000)
    parser.add_argument("--frames_per_sample", type=int, default=15)
    parser.add_argument("--video_dir", type=str, default="./videos")
    parser.add_argument("--overwrite_existing_dset", type=bool, default=True)
    args = parser.parse_args()
    main(args)
