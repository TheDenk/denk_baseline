import os
import glob

import torch
import pandas as pd
from tqdm import tqdm

from video_processor import VideoProcessor
from tta import five_crops, ten_crops, one_flip
from models.efficienx3d import EfficientX3d


common_params = {
    'dataset_dir': 'dataset',
    # 'dataset_dir': '/home/user/datasets/slovo/dataset_18',
    'output_file': 'predicts.csv',
    'device': 'cuda:0',
    'min_class_treshold': 0.1,
    'use_tta': True,
}

model_params = {
    'num_classes': 1001,
    'coef': 5.0,
    'ckpt_path': './efficientx3d-finetuned-0.686.ckpt',
}

video_processor_params = {
    'min_side': 256, 
    'sample_size': 256, 
    'sample_n_frames': 32, 
    'sample_stride': 2, 
    'batch_stride': 16,
}

if __name__ == "__main__":
    videos = glob.glob(os.path.join(common_params['dataset_dir'], '*.mp4'))

    model = EfficientX3d(**model_params)
    model = model.eval().to(device=common_params['device'])
    
    names = []
    predicts = []
    for video_path in tqdm(videos):
        name = os.path.basename(video_path).replace('.mp4', '')
        names.append(name)
        video_processor = VideoProcessor(video_path=video_path, **video_processor_params)

        # --- START --- VIDEO BATCH ---
        indexes = []
        values = []
        batch_frames = video_processor.get_video_batch(return_full=False, repeat_last=False)
        with torch.no_grad():
            for frames in batch_frames:
                if common_params['use_tta']:
                    frames = frames.unsqueeze(0)
                    # frames = five_crops(frames, crop_h=224, crop_w=224)
                    frames = one_flip(frames)
                    predicted = model(frames.to(common_params['device'])).sum(dim=0).softmax(dim=0)
                else:
                    predicted = model(frames.to(common_params['device']).unsqueeze(0)).softmax(dim=1).squeeze(0)

                indexes.append(predicted.argmax().cpu().item())
                values.append(predicted.amax().cpu().item())

        ## Last no event
        # predicted_class = 1000
        # for index, value in zip(indexes, values):
        #     if (index != 1000) and (value >= common_params['min_class_treshold']):
        #         predicted_class = index

        ## Last max value and no event
        # predicted_class = 1000
        # max_value = values[0]
        # for index, value in zip(indexes, values):
        #     if (index != 1000) and (value >= max_value):
        #         predicted_class = index
        #         max_value = value

        ## Most common value
        from collections import Counter

        t_indexes = [1000]
        for index, value in zip(indexes, values):
            if value >= common_params['min_class_treshold']:
                t_indexes.append(index)

        predicted_class = 1000
        for (index, count) in Counter(t_indexes).most_common(2):
            if index != 1000:
                predicted_class = index
                break
        # --- END --- VIDEO BATCH ---

        # --- START --- FIRST FRAMES ---
        # frames = video_processor.get_video(start_frame=0, return_full=False)
        # if common_params['use_tta']:
        #     frames = frames.unsqueeze(0)
        #     frames = five_crops(frames, crop_h=224, crop_w=224)
        #     with torch.no_grad():
        #         predicted = model(frames.to(common_params['device'])).sum(dim=0).softmax(dim=0)

        # else:
        #     with torch.no_grad():
        #         predicted = model(frames.to(common_params['device']).unsqueeze(0)).squeeze(0)
        # predicted_class = int(predicted.argmax().cpu().item())
        # --- END --- FIRST FRAMES ---
        predicts.append(predicted_class)

        del video_processor

    result_df = pd.DataFrame.from_dict({"attachment_id":names, "class_indx":predicts})
    
    result_df.to_csv(common_params['output_file'], sep="\t", index=False)
    

