import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch
import numpy as np

from src.datamodules.components.dataset import BaseDataset

json_path = ' ' #path to preprocessed json file
class TextClassificationDataset(BaseDataset):
    def __init__(
        self,
        json_path: json_path,
        txt_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        shuffle_on_load: bool = True,
        label_type: str = "torch.LongTensor",
        **kwargs: Any,
    ) -> None:
        """TextClassificationDataset.

        Args:
            json_path (str, optional): Path to annotation JSON file.
            txt_path (str, optional): Path to annotation text file.
            transforms (Callable): Transforms.
            shuffle_on_load (bool): Deterministically shuffle the dataset on load
                to avoid the case when Dataset slice contains only one class due to
                annotations dict keys order. Default to True.
            label_type (str): Label torch.tensor type. Default to torch.FloatTensor.
            kwargs (Any): Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms)
        
        if (json_path and txt_path) or (not json_path and not txt_path):
            raise ValueError("Requires json_path or txt_path, but not both.")
        elif json_path:
            json_path = Path(json_path)
            if not json_path.is_file():
                raise RuntimeError(f"'{json_path}' must be a file.")
            with open(json_path) as json_file:
                self.annotation = json.load(json_file)
        else:
            txt_path = Path(txt_path)
            if not txt_path.is_file():
                raise RuntimeError(f"'{txt_path}' must be a file.")
            self.annotation = {}
            with open(txt_path) as txt_file:
                for line in txt_file:
                    text, label = line.split("\t")
                    self.annotation[text] = label

        self.keys = list(self.annotation)
        if shuffle_on_load:
            random.shuffle(self.keys)

        self.label_type = label_type

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        text = key  # In this example, we assume the key itself is the text (modify as needed)
        text = self._process_text(text)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        return {"text": text, "label": label}

    def get_weights(self) -> List[float]:
        label_list = [self.annotation[key] for key in self.keys]
        weights = 1.0 / np.bincount(label_list)
        return weights.tolist()

    def _process_text(self, text: str) -> str:
        #data is already pre processed 
        return text

