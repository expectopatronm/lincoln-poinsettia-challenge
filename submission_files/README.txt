Shankar Kumar

This is my submission to the 'Find the bracts (plant heads) challenge.

The file 'results.json' contains the resulting predictions in the specified COCO format ([{"image_id": 0, "category_id": 1, "bbox": [338.2, 425.8, 27.7, 28.7], "score": 0.99}]...)

If you want to regenerate the 'results.json' file:
Execution instructions:
1. run 'pip install -qr requirements.txt' to install all necessary packages and     libraries
2. run 'python inference.py --source PATH_TO_TEST_DATASET' to generate the .json    results file
   eg: 'python inference.py --source ../test'


Note: Only the Inference files are submitted since training would anyway not be possible without the augmented and corrected images I used numbering in the thousands. Augmentation was done on an online software as a service platform, so this cannot be made available. Hence no training code/script.
With the 'inference.py' script my 'results.json' file can be regenerated for verifification.