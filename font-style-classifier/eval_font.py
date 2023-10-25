from common import *
import pandas as pd
from PIL import Image
from time import time
import torchvision.transforms.functional as F
from db_retrieve import retrieve, aggregate_retrieved_results

# Evaluate the font classification performance: given a character of the testing set, we evaluate whether the character
# is correctly classified as belonging to its font or not.

# Load the test set
test_df = pd.read_csv(FSC_TEST_SET_CSV)

last_logged_at = 0

font_classification_results = {}  # How many times a font was correctly classified

for i, row in test_df.iterrows():
    img_path = os.path.join(FSC_DATASET_DIR, row["filename"])
    img = Image.open(img_path)
    img = (F.pil_to_tensor(img) / 255.0).float()

    retrieved_results = retrieve(img, 10)

    # We filter out the retrieved samples that are from test set (e.g. the query sample would be returned as the most
    # similar!)

    def is_from_test_set(sample_payload: dict) -> bool:
        return (
                (test_df['font'] == sample_payload['font']) &
                (test_df['char'] == sample_payload['char']) &
                (test_df['is_bold'] == sample_payload['is_bold']) &
                (test_df['filename'] == sample_payload['filename'])
        ).any()

    retrieved_results = [x for x in retrieved_results if not is_from_test_set(x['payload'])]

    true_font = row['font']
    predicted_font = aggregate_retrieved_results(retrieved_results)

    if true_font not in font_classification_results:
        font_classification_results[true_font] = {'correct': 0, 'total': 0}

    font_classification_results[true_font]['correct'] += (true_font == predicted_font)
    font_classification_results[true_font]['total'] += 1

    if (time() - last_logged_at) >= 1.0:
        print("Processed %d/%d test set items..." % (int(i), len(test_df)))
        last_logged_at = time()


total_correct_count = sum(map(lambda x: x['correct'], font_classification_results.values()))
total_count = len(test_df)

for font, value in font_classification_results.items():
    print("Font: \"%s\", Correct: %d, Total: %d" % (font, value['correct'], value['total']))

print("Total correct: %d/%d" % (total_correct_count, total_count))
