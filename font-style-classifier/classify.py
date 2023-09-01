import torch
import torchvision.transforms.functional as F
import torchvision
from model.model import FSC_Encoder
from qdrant_client import QdrantClient
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import test_env


FSC_DB_PATH="/home/rutayisire/projects/dataset-retriever/font-style-classifier/.fsc-db"
FSC_DB_COLLECTION_NAME="embeddings"

FSC_DATASET_CSV="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/"

print("Loading the model...")
model = FSC_Encoder()
model.load_checkpoint("/home/rutayisire/projects/dataset-retriever/font-style-classifier/model/latest-checkpoint.pt")
model = model.cuda()

print("Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH)

print("Init done!")

class ClassifyFontStyle:
    def __init__(self, char_list: list):
        """
        Args:
            char_list:
                A list of tuples (char_image, char), where char_image is a (1, 32, 32) PyTorch tensor and char
                is the corresponding character.
        """

        self.char_list = char_list

        self.model_input = torch.stack([char_img for char_img, _ in self.char_list])
        self.model_input = self.model_input.cuda()

        self.dataset = pd.read_csv(FSC_DATASET_CSV)


    def _calc_embeddings(self):
        self.embeddings = model(self.model_input)


    def _find_nearest_fonts(self):
        self.nearest_fonts = []
        for embedding in self.embeddings:
            nn = db_client.search(
                collection_name=FSC_DB_COLLECTION_NAME,
                query_vector=embedding.tolist(),
                with_vectors=True,
                limit=100
                )
            
            # TODO 
            self.nearest_fonts.append(nn[0].payload['font'])


    def _load_dataset_image(self, filename: str):
        img_path = path.join(FSC_DATASET_DIR, filename)
        img = Image.open(img_path) 
        return F.to_tensor(img)
    

    def _load_styled_chars(self):
        self.styled_chars = []

        for (_, char), nearest_font in zip(self.char_list, self.nearest_fonts):
            q = self.dataset
            
            # Given the nearest font, load the (regular, italic, bold) versions of the character
            q = q[(q['font'] == nearest_font) & (q['char'] == char)]

            regular_df = q[(~q['is_italic']) & (~q['is_bold'])]  # Regular (not italic nor bold)
            italic_df = q[q['is_italic']]
            bold_df = q[q['is_bold']]

            regular_img = None
            if not regular_df.empty:
                regular_img = self._load_dataset_image(
                    path.join(FSC_DATASET_DIR, regular_df.iloc[0]['filename'])
                    )
            
            italic_img = None
            if not italic_df.empty:
                italic_img = self._load_dataset_image(
                    path.join(FSC_DATASET_DIR, italic_df.iloc[0]['filename'])
                    )

            bold_img = None
            if not bold_df.empty:
                bold_img = self._load_dataset_image(
                    path.join(FSC_DATASET_DIR, bold_df.iloc[0]['filename'])
                    )

            self.styled_chars.append((regular_img, italic_img, bold_img))
    

    def __call__(self):
        print("Calculating embeddings...")
        self._calc_embeddings()

        print("Searching for nearest fonts...")
        self._find_nearest_fonts()

        print("Loading styled chars...")
        self._load_styled_chars()


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_doc = test_env.sample_document()

    char_list = []
    for i in range(128):
        _, _, char_img, char_info = test_doc.sample_char()
        char_list.append((char_img, char_info['char']))
    
    classify = ClassifyFontStyle(char_list)
    classify()

    # Display a grid showing the first num_rows characters
    grid = []
    num_rows = 10

    for (char_img, char), (regular_img, italic_img, bold_img) in zip(char_list, classify.styled_chars):
        grid += [
            char_img,
            regular_img if regular_img != None else torch.zeros((1, 32, 32)),
            italic_img if italic_img != None else torch.zeros((1, 32, 32)),
            bold_img if bold_img != None else torch.zeros((1, 32, 32)),
            ]
        
        num_rows -= 1
        if num_rows == 0:
            break

    grid = torchvision.utils.make_grid(grid, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

