from db_retrieve import retrieve, aggregate_retrieved_results
import matplotlib.pyplot as plt
from dataset import *
import random
import itertools

# Evaluate the font classification performance: given a character of the testing set, we evaluate whether the character
# is correctly classified as belonging to its font or not.


def _show_retrieved_samples(test_word):
    num_retrieved_samples = 10

    def _display(highlight_font=None, highlight_color='green'):
        fig, axs = plt.subplots(num_retrieved_samples + 1, 4, figsize=(8, 10))

        for ax in itertools.chain(*axs):
            ax.axis('off')

        def _char_description(char, d=None):
            font_style = ["r", "b", "i", "bi"][char["is_italic"] * 2 + char["is_bold"]]
            out = f"{char['font']} ({font_style})"
            if d is not None:
                out += "\n"
                out += "{:.6f}".format(d)
            return out

        word_ret_samples = []

        if highlight_font is None:
            highlight_font = test_word[0]['font']

        for i, test_char in enumerate(test_word):
            axs[0, i].imshow(plt.imread(get_dataset_image_path(test_char['filename'])), cmap='gray')

            text = _char_description(test_char)
            axs[0, i].text(0.5, -0.1, text,
                           transform=axs[0, i].transAxes,
                           ha='center', va='top',
                           c='blue'
                           )

            test_char_img = load_dataset_image(test_char['filename'])

            ret_samples = retrieve(test_char_img, num_retrieved_samples, skip_first=True)
            word_ret_samples.append(ret_samples)

            for j, ret_sample in enumerate(ret_samples):
                ret_sample_payload = ret_sample['payload']

                axs[j + 1, i].imshow(plt.imread(get_dataset_image_path(ret_sample_payload['filename'])), cmap='gray')

                text = _char_description(ret_sample_payload, d=ret_sample['distance'])
                text_color = highlight_color if ret_sample_payload['font'] == highlight_font else 'red'

                axs[j + 1, i].text(0.5, -0.1, text,
                                   transform=axs[j + 1, i].transAxes,
                                   ha='center', va='top',
                                   c=text_color
                                   )

        plt.tight_layout()
        plt.show()

        return word_ret_samples

    # The first time display test word characters along with retrieval results.
    # Highlight characters having the same font of the test word
    word_ret_samples = _display()

    # Aggregate the retrival results (for the whole word) to obtain the classification result.
    # It's very imprecise but at least we get a similar looking font
    best_font = aggregate_retrieved_results(word_ret_samples)

    # Now display the UI to highlight characters of the classification result
    _display(highlight_font=best_font, highlight_color='blue')


def _main():
    fonts = list(get_complete_fonts())
    random.shuffle(fonts)

    test_df = pd.read_csv(FSC_TEST_SET_CSV)

    while len(fonts) > 0:
        font = fonts.pop()

        # Returns a list of rows from the test set having the given font.
        # It returns 4 rows (because every font has 4 test characters); these make up the word for which we want to classify the font
        test_word = list(test_df[test_df['font'] == font].T.to_dict().values())

        _show_retrieved_samples(test_word)  # Show the retrieval results in a UI


if __name__ == "__main__":
    _main()
