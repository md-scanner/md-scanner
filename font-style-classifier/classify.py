import torch
import torchvision.transforms.functional as F
import torchvision
from model.model import FSC_Encoder
from qdrant_client import QdrantClient
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from PIL import Image
import time
import test_env


FSC_DB_PATH="/home/carmine/gits/dataset-retriever/font-style-classifier/.fsc-db"
FSC_DB_COLLECTION_NAME="embeddings"

<<<<<<< HEAD
FSC_DATASET_CSV="/home/carmine/gits/dataset-retriever/font-style-classifier/dataset/dataset.csv"
FSC_DATASET_DIR="/home/carmine/gits/dataset-retriever/font-style-classifier/dataset/"
FSC_CHECKPOINT_FILE="/home/carmine/gits/dataset-retriever/font-style-classifier/checkpoint-20230904150346.pt"
=======
FSC_DATASET_CSV="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/"
FSC_CHECKPOINT_FILE="/home/rutayisire/projects/dataset-retriever/font-style-classifier/model/latest-checkpoint-verytiny.pt"
>>>>>>> 17480e0 (fsc: Minor updates)

print("Loading the model...")
model = FSC_Encoder()
model.load_checkpoint(FSC_CHECKPOINT_FILE)
model = model.cuda()

print("Initializing the DB... ", end="")
db_client = QdrantClient(path=FSC_DB_PATH)
el_count = db_client.count(collection_name=FSC_DB_COLLECTION_NAME).count
print(f"Elements: {el_count}")

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
        self.batch_size = len(char_list)

        self.model_input = torch.stack([char_img for char_img, _ in self.char_list])
        self.model_input = self.model_input.cuda()

        self.dataset = pd.read_csv(FSC_DATASET_CSV)


    def _calc_embeddings(self):
        self.embeddings = model(self.model_input)


    def _find_nearest_fonts(self):
        self.nearest_fonts = []
        for embedding in self.embeddings:
            query_result = db_client.search(
                collection_name=FSC_DB_COLLECTION_NAME,
                query_vector=embedding.tolist(),
                with_vectors=True,
                limit=10
                )
            
            near_fonts = [entry.payload['font'] for entry in query_result]
            nearest_font = max(set(near_fonts), key=near_fonts.count)
            #print(f"Nearest font: {nearest_font}, Count: {near_fonts.count(nearest_font)}")

            self.nearest_fonts.append(nearest_font)


    def _load_dataset_image(self, filename: str):
        img_path = path.join(FSC_DATASET_DIR, filename)
        img = Image.open(img_path)
    
        t = F.to_tensor(img)
        t = torch.floor(t)
        return t


    def _load_styled_chars(self):
        self.styled_chars = []

        for (_, char), nearest_font in zip(self.char_list, self.nearest_fonts):
            q = self.dataset
            
            # Given the nearest font, load the (regular, italic, bold) versions of the character
            q = q[(q['font'] == nearest_font) & (q['char'] == char)]

            # We don't have that char in the dataset (e.g. it's a special character such as '-_")
            if q.empty:
                self.styled_chars.append(None)
                continue

            regular_df = q[(~q['is_italic']) & (~q['is_bold'])]  # Regular (not italic nor bold)
            italic_df = q[q['is_italic']]
            bold_df = q[q['is_bold']]

            # IMPORTANT: the DB must only have fonts with all regular, italic and bold styles
            assert not regular_df.empty
            assert not italic_df.empty
            assert not bold_df.empty

            regular_img = self._load_dataset_image(
                path.join(FSC_DATASET_DIR, regular_df.iloc[0]['filename'])
                )
            
            italic_img = self._load_dataset_image(
                path.join(FSC_DATASET_DIR, italic_df.iloc[0]['filename'])
                )

            bold_img = self._load_dataset_image(
                path.join(FSC_DATASET_DIR, bold_df.iloc[0]['filename'])
                )

            self.styled_chars.append((regular_img, italic_img, bold_img))
    

    def _classify(self):
        # List of indices of the characters that can be classified (i.e. that have a regular/italic/bold version found in the dataset)
        classifiable_indices = [i for i, entry in enumerate(self.styled_chars) if entry != None]

        original = torch.stack([self.char_list[i][0] for i in classifiable_indices])
        
        regular = torch.stack([self.styled_chars[i][0] for i in classifiable_indices])
        bold = torch.stack([self.styled_chars[i][1] for i in classifiable_indices])
        italic = torch.stack([self.styled_chars[i][2] for i in classifiable_indices])

        t = torch.stack([original, regular, bold, italic])
        t = torch.swapaxes(t, 0, 1)
        # t.shape: (B, 4, 1, 32, 32)

        a = t[:,:1] # (B, 1, 1, 32, 32)
        b = t[:,1:] # (B, 3, 1, 32, 32)

        d = torch.sum(torch.abs(a - b), dim=(2,3,4))
        # d.shape: (B, 3)

        nearest_indices = torch.argmin(d, dim=1).tolist()  # (B,)
        
        self.style_indices = [None] * self.batch_size
        for i, style_idx in enumerate(nearest_indices):
            self.style_indices[classifiable_indices[i]] = style_idx


    def __call__(self):
        #print(f"Calculating embeddings...; Batch size: {self.batch_size}, ", end="")
        st = time.time()
        self._calc_embeddings()
        dt = time.time() - st
        #print(f"DT: {dt:.3f}")

        #print(f"Searching for NN...; Batch size: {self.batch_size}, ", end="")
        st = time.time()
        self._find_nearest_fonts()
        dt = time.time() - st
        #print(f"DT: {dt:.3f}")

        #print(f"Loading styled characters...; Batch size: {self.batch_size}, ", end="")
        st = time.time()
        self._load_styled_chars()
        dt = time.time() - st
        #print(f"DT: {dt:.3f}")

        #print(f"Performing style classification...; Batch size: {self.batch_size}, ", end="")
        st = time.time()
        self._classify()
        dt = time.time() - st
        #print(f"DT: {dt:.3f}")

        self.result = self.style_indices

        #print("Done!")

        return self.result


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_doc = test_env.sample_document()

    # Show the sampled document
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(test_doc.img.permute(2, 1, 0), cmap='gray', vmin=0, vmax=255)
    axs[0].axis('off')

    axs[1].imshow(test_doc.bin_img.permute(2, 1, 0), cmap='gray', vmin=0, vmax=255)
    axs[1].axis('off')

    plt.show()

    ##
    num_samples = 10

    while True:
        # Draw num_samples characters from the document 
        st = time.time()
        char_list = []
        for i in range(num_samples):
            _, _, char_img, char_info = test_doc.sample_char()
            char_list.append((char_img, char_info['char']))
        dt = time.time() - st
        print(f"Drawn {num_samples} characters; dt: {dt:.3f}")
        
        # Classify the drawn samples
        st = time.time()
        classify = ClassifyFontStyle(char_list)
        classify()
        dt = time.time() - st
        print(f"Classified; dt: {dt:.3f}")

        # Create a grid listing the drawn characters with their matchings
        st = time.time()
        grid = []
        for \
            (char_img, char), \
            nearest_font, \
            styled_images \
        in zip(char_list, classify.nearest_fonts, classify.styled_chars):
            grid += [
                char_img,
                styled_images[0] if styled_images != None else torch.zeros((1, 32, 32)),
                styled_images[1] if styled_images != None else torch.zeros((1, 32, 32)),
                styled_images[2] if styled_images != None else torch.zeros((1, 32, 32)),
                ]
        dt = time.time() - st
        print(f"Created display grid; dt: {dt:.3f}")

        # Show the plot
        fig, ax = plt.subplots(figsize=(8, 14))
        
        grid = torchvision.utils.make_grid(grid, nrow=4)
        ax.imshow(grid.permute(1, 2, 0))

        for i, style_idx in enumerate(classify.style_indices):
            ax.add_patch(plt.Rectangle(((style_idx + 1) * 32, i * 32), 32, 32, fill=False, edgecolor='lime', linewidth=1))

        ax.axis('off')
        plt.show()

