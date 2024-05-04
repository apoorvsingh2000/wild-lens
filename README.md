## WildLens


This project aims to develop an innovative algorithm for detecting and classifying wildlife in their natural habitats.

### Steps to train

1. Download the dataset (~ 120 GB)

    ```angular2html
    kaggle competitions download -c iwildcam-2020-fgvc7
    ```

2. Merge all the required data into a single csv file

    ```angular2html
    python join_data.py
    ```

3. Train the model

    ```angular2html
   python train_model.py --start 0 --num-epochs 25 --batch-size 64 --model-name "vit_full"
   ```
