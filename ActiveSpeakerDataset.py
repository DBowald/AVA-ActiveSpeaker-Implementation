from torch.utils.data import Dataset
import glob
import os
import cv2
import pandas as pd
import wget

#class ActiveSpeakerDataset(Dataset):

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


def create_samples(csv_path, vid_path="./videos", temporal_window=2, overlap=1, min_frames=5, sample_freq=20):
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
        break
    return samples

if __name__ == "__main__":
    csv_path = "./data/ava_activespeaker_test_v1.0"
    samples = create_samples(csv_path)
    print(len(samples))

