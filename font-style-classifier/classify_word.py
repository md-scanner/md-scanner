from encoder.model import model
from common import *
from db_retrieve import retrieve_with_embedding, aggregate_retrieved_results
from dataset import load_dataset_image, filter_dataset


# We assume word characters to all share the same font, in order to get better results

def classify_word_font(word) -> str:
    """ Given a word, classifies its font by using DB retrieval.
    Args:
        word:
            A list of tuples (image, char). Where `image` is a (1, 32, 32) tensor and `char` is the represented char.
    Returns:
        The font considered to be the best match.
    """

    batch = torch.stack([x for x, _ in word])
    embeddings = model(batch)

    num_retrieved_samples = 5
    word_retrieve_results = []
    for embedding in embeddings:
        word_retrieve_results.append(
            retrieve_with_embedding(embedding, count=num_retrieved_samples)
        )

    best_font = aggregate_retrieved_results(word_retrieve_results)
    return best_font


def classify_char_style(char, font: str) -> int:
    """ Given a char, classifies its font style (regular, bold, italic).
    Args:
        char: A tuple (image, char);
            the first entry is the character (string), the second entry is a (1, 32, 32) tensor.
    """

    # Take the same character in different styles for the classified font (regular, bold, italic)
    # TODO Use filter_dataset
    regular_img = load_dataset_image(filter_dataset(font=font, font_style=0, char=char[1]).iloc[0]['filename'])
    bold_img = load_dataset_image(filter_dataset(font=font, font_style=1, char=char[1]).iloc[0]['filename'])
    italic_img = load_dataset_image(filter_dataset(font=font, font_style=2, char=char[1]).iloc[0]['filename'])

    # Compute the distance of the character with the three versions and the final classification is the one whose
    # SAD (Sum of Absolute Distance) is less
    a = torch.unsqueeze(char[0], dim=0)
    b = torch.stack([regular_img, bold_img, italic_img])

    d = torch.sum(torch.abs(a - b), dim=(1, 2, 3))
    return torch.argmin(d, dim=0).item()


def classify_word_style(word) -> int:
    """ Given a word, classifies its font style (regular, bold, italic).
    Args:
        word:
            A list of tuples (image, char). Where `image` is a (1, 32, 32) tensor and `char` is the represented char.
    Returns:
        0 if regular, 1 if bold, 2 if italic. Note: bold/italic classification not provided.
    """

    word_txt = "".join([x for _, x in word])
    font = classify_word_font(word)
    # print(f"[classify_word] Word \"{word_txt}\", best font: \"{font}\"")

    font_styles = []
    for char_tuple in word:
        font_style = classify_char_style(char_tuple, font)
        font_styles.append(font_style)

        # print(f"[classify_word] \"{char_tuple[1]}\" -> {font_style_str(font_style)}")

    # Take the max voted font style as the final classification result
    best_font_style = max(set(font_styles), key=font_styles.count)

    # print(f"[classify_word] Word \"{word_txt}\" -> {font_style_str(best_font_style)}")

    return best_font_style
