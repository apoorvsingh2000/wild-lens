import json
import os
import time
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED = True

IMAGE_DIR = './iwildcam-2020-fgvc7/train'


def check_validity(file_path):
    try:
        Image.open(file_path)
        return os.path.isfile(file_path)
    except:
        return False


def main():
    with open('./iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
        labels = pd.DataFrame(data['annotations'])
        images = pd.DataFrame(data["images"])

        annotations_df = labels[['image_id', 'category_id']]
        images_df = images[['id', 'file_name']].rename(columns={'id': 'image_id'})
        df = pd.merge(annotations_df, images_df, on='image_id')

        df['image_path'] = df['file_name'].apply(lambda x: IMAGE_DIR + '/' + x)
        df['is_valid'] = df['image_path'].apply(check_validity)
        df = df[df['is_valid']]
        df = df.drop(columns=['is_valid', 'image_path'])

        df.to_csv('./wild-lens.csv', index=False)


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f'Finished in {round(end_time - start_time, 2)} seconds')
