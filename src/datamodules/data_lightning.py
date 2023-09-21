import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QGenModel(pl.LightningModule):
    def __init__(
        self,
        generator_name_or_path="doc2query/msmarco-t5-base-v1",
        ques_per_passage=3,
        bsz=32,
        qgen_prefix="QGen",
        save=True,
        save_after=10000,
        data_path=None
    ):
        super().__init__()

        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name_or_path)
        self.ques_per_passage = ques_per_passage
        self.bsz = bsz
        self.qgen_prefix = qgen_prefix
        self.save = save
        self.save_after = save_after
        self.data_path = data_path

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Create the Lightning module.
    model = QGenModel(data_path=args.data_path)

    # Create the Trainer.
    logger = wandb("logs")
    trainer = pl.Trainer(logger=logger)

    # Train the model.
    trainer.fit(model, model.dataset)
