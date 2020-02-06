from ActiveSpeakerDataset import create_samples
import cv2
import random
import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser

def main(args):
    #csv_dir = "./data/ava_activespeaker_test_v1.0"
    #out_dir = "./data/ava_activespeaker_samples/test"

    # n_samples = 4000
    # frames_per_sample = 15

    if(args.overwrite_existing_dset):
        pass

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
                cap_path = os.path.join(args.video_dir, youtube_id + "*")
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
                    curr_frame = each_image[1]
                    top_left = each_image[2]
                    bot_right = each_image[3]
                    if(curr_frame != prev_frame+1):
                        vcap.set(1, curr_frame)
                    ret, frame = vcap.read()
                    roi = frame[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
                    prev_frame = curr_frame
                    save_path = os.path.join(out_dir, dirname, "{}-{}.png".format(curr_frame, label))
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
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--frames_per_sample", type=int, default=15)
    parser.add_argument("--video_dir", type=str, default="./videos")
    parser.add_argument("--overwrite_existing_dset", type=bool, default=True)
    args = parser.parse_args()
    main(args)
