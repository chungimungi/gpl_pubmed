from pathlib import Path
from typing import Any, Callable, Optional
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None
    ) -> None:
        """TextDataset.

        Args:
            transforms (Callable): Text preprocessing transforms.
        """

        self.transforms = transforms

    def _read_text_(self, text: Any) -> str:
        """Read text from source.

        Args:
            text (Any): Text source. Could be str, Path, or bytes.

        Returns:
            str: Loaded text.
        """

        if isinstance(text, (str, Path)):
            with open(text, 'r', encoding='utf-8') as file:
                text = file.read()
        elif isinstance(text, bytes):
            text = text.decode('utf-8')
        return text

    def _process_text_(self, text: str) -> torch.Tensor:
        """Process text, including text preprocessing transforms, etc.

        Args:
            text (str): Text data.

        Returns:
            torch.Tensor: Text prepared for dataloader.
        """
        if self.transforms:
            text = self.transforms(text)
        return text

    def __getitem__(self, index: int) -> Any:
        # Implement this method to load and process text data based on your specific dataset structure.
        # For example, you might load JSON files, preprocess them, and return the text data.
        raise NotImplementedError()

    def __len__(self) -> int:
        # Implement this method to return the total number of text samples in your dataset.
        raise NotImplementedError()
