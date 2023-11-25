# Native imports
import os

# Installed imports
import faiss
from PIL import Image
from datasets import Dataset
import matplotlib.pyplot as plt

# Local imports
from .EfficientNet import get_embeddings


TMP_PATH = "tmp/index_file.index"


class ImageIndexer:
    DIM = 2560  # Default value for our EfficientNet

    def __init__(self, index_path: str, dataset: Dataset, n_items: int = None, reindex: bool = False):
        assert isinstance(index_path, str) and index_path.endswith(".index")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        self.dataset = dataset
        self.n_items = n_items
        self.index_path = index_path
        self.index = self._create_index() if not os.path.isfile(index_path) or reindex else self._load_index()

    def _create_index(self):
        index = faiss.IndexFlatL2(self.DIM)
        images = self.dataset['train']['image'][:self.n_items]
        embedds = get_embeddings(images)
        index.add(embedds)
        faiss.write_index(index, self.index_path)
        self.embedds = embedds
        return index

    def _load_index(self):
        try:
            return faiss.read_index(self.index_path)
        except Exception as e:
            raise ValueError("Please define a valid path or pass the dataset to the class in order to create a new one") from e

    def search_image(self, image: Image, top_k: int = 1):
        embedd = get_embeddings([image])
        self.embedd = embedd
        distances, indices = self.index.search(embedd[0].reshape(1, -1), top_k)
        return distances[0], [self.dataset['train']['image'][i] for i in indices[0]]

    def search_image_with_plot(self, image: Image, top_k: int = 1, figsize: tuple = (2, 10)):
        distances, images = self.search_image(image, top_k)
        n_images = len(images)

        fig, axes = plt.subplots(n_images + 1, 1, figsize=figsize)
        axes[0].imshow(image)
        for i, (dist, im) in enumerate(zip(distances, images)):
            axes[i+1].imshow(im)
            axes[i+1].set_title(f"Score:{dist:0f}")
        plt.show()

        return distances, images