import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_dataset_adapter(adapter_type):
    if adapter_type == "huggingface_dataset":
        return HuggingFaceDatasetAdapter()
    else:
        raise ValueError(f"Adapter '{adapter_type}' non riconosciuto.")


class HuggingFaceDatasetAdapter:
    """
    Adattatore per dataset caricati dalla HuggingFace Hub.
    """

    def load_dataset_from_name(self, dataset_name, tokenizer, resolution, args):
        from datasets import load_dataset

        # Carica il dataset dalla HuggingFace Hub
        dataset = load_dataset(
            dataset_name,
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
            trust_remote_code=True,
        )

        column_names = dataset["train"].column_names

        # Determina le colonne di immagini e caption
        image_column = args.image_column if args.image_column else "image"
        caption_column = args.caption_column if args.caption_column else "text"

        # Preprocessing delle immagini e delle caption
        def preprocess(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            transforms_list = transforms.Compose(
                [
                    transforms.Resize(
                        resolution, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(resolution)
                        if args.center_crop
                        else transforms.RandomCrop(resolution)
                    ),
                    (
                        transforms.RandomHorizontalFlip()
                        if args.random_flip
                        else transforms.Lambda(lambda x: x)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            examples["pixel_values"] = [transforms_list(image) for image in images]
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    captions.append(random.choice(caption))
                else:
                    raise ValueError(
                        "Il campo caption deve essere una stringa o una lista di stringhe."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            examples["input_ids"] = inputs.input_ids.squeeze()
            return examples

        # Applica il preprocessing
        train_dataset = dataset["train"].with_transform(preprocess)
        return train_dataset

    def load_dataset(self, dataset_path, tokenizer, resolution, args):
        # Puoi implementare questa funzione per gestire dataset locali se necessario
        pass
