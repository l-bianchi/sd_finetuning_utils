import os
import json
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def get_dataset_adapter(adapter_type):
    if adapter_type == "folder_with_captions":
        return FolderWithCaptionsAdapter()
    elif adapter_type == "images_with_text_files":
        return ImagesWithTextFilesAdapter()
    else:
        raise ValueError(f"Adapter '{adapter_type}' non riconosciuto.")


class FolderWithCaptionsAdapter:
    """
    Adattatore per dataset organizzati come una cartella con immagini e un file metadata.jsonl.
    """

    def load_dataset(self, dataset_path, tokenizer, resolution):
        # Carica il file metadata.jsonl
        metadata_file = os.path.join(dataset_path, "metadata.jsonl")
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Prepara una lista di tuple (immagine, caption)
        data = []
        for line in lines:
            item = json.loads(line)
            image_path = os.path.join(dataset_path, item["file_name"])
            caption = item["text"]
            data.append((image_path, caption))

        # Crea il dataset personalizzato
        return CustomImageDataset(data, tokenizer, resolution)


class ImagesWithTextFilesAdapter:
    """
    Adattatore per dataset dove ogni immagine ha un file di testo associato con la caption.
    """

    def load_dataset(self, dataset_path, tokenizer, resolution):
        # Ottieni la lista di tutti i file immagine
        image_files = [
            f
            for f in os.listdir(dataset_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        data = []
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            # Assumi che il file di testo abbia lo stesso nome ma con estensione .txt
            text_file = os.path.splitext(image_file)[0] + ".txt"
            text_path = os.path.join(dataset_path, text_file)
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                data.append((image_path, caption))
            else:
                print(
                    f"Attenzione: Nessuna caption trovata per l'immagine {image_file}"
                )

        # Crea il dataset personalizzato
        return CustomImageDataset(data, tokenizer, resolution)


class CustomImageDataset(Dataset):
    def __init__(self, data, tokenizer, resolution):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        example = {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(),
        }
        return example
