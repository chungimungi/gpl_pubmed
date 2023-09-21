import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import wandb
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@hydra.main(config_path="example.yaml")
class QGenModel(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.generator = AutoModelForSeq2SeqLM.from_pretrained(cfg.generator_name_or_path)
        self.ques_per_passage = cfg.ques_per_passage
        self.bsz = cfg.bsz
        self.qgen_prefix = cfg.qgen_prefix
        self.save = cfg.save
        self.save_after = cfg.save_after
        self.data_path = cfg.data_path

        # Load the dataset
        self.dataset = load_dataset(self.data_path)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        # Generate queries for the current batch.
        queries = self.generator.generate(
            batch['text'],
            num_return_sequences=self.ques_per_passage,
            max_length=64,
            prefix=self.qgen_prefix,
        )

        # Save the generated queries to a file.
        if self.save and batch_idx % self.save_after == 0:
            self.save_queries(queries, f'queries_{batch_idx}.jsonl')

        return queries

    def save_queries(self, queries, filename):
        with open(filename, 'w', encoding='utf-8') as fOut:
            for query in queries:
                json.dump(query, fOut)
                fOut.write('\n')

if __name__ == "__main__":
    hydra.run(QGenModel)
