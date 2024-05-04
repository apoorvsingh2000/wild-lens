import json
import pandas as pd


def main():
    with open('./iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
        labels = pd.DataFrame(data['annotations'])
        images = pd.DataFrame(data["images"])

        annotations_df = labels[['image_id', 'category_id']]
        images_df = images[['id', 'file_name']].rename(columns={'id': 'image_id'})
        df = pd.merge(annotations_df, images_df, on='image_id')

        df.to_csv('./wild-lens.csv', index=False)


if __name__ == '__main__':
    main()
