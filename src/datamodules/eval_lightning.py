import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class GPLRetrieval(pl.LightningModule):
    def __init__(self, model, batch_size=128):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        corpus, queries, qrels = batch

        # Retrieve the top k documents for each query.
        results = self.model.retrieve(corpus, queries)

        # Calculate the loss.
        loss = self.model.loss(results, qrels)

        # Return the loss.
        return loss

    def validation_step(self, batch, batch_idx):
        corpus, queries, qrels = batch

        # Retrieve the top k documents for each query.
        results = self.model.retrieve(corpus, queries)

        # Calculate the NDCG@2, recall@1, and precision@1.
        ndcg2 = EvaluateRetrieval.ndcg_at_k(results, qrels, k=2)
        recall1 = EvaluateRetrieval.recall_at_k(results, qrels, k=1)
        precision1 = EvaluateRetrieval.precision_at_k(results, qrels, k=1)

        # Return the NDCG@2, recall@1, and precision@1.
        return ndcg2, recall1, precision1

    def test_step(self, batch, batch_idx):
        corpus, queries, qrels = batch

        # Retrieve the top k documents for each query.
        results = self.model.retrieve(corpus, queries)

        # Calculate the NDCG@2, recall@1, and precision@1.
        ndcg2 = EvaluateRetrieval.ndcg_at_k(results, qrels, k=2)
        recall1 = EvaluateRetrieval.recall_at_k(results, qrels, k=1)
        precision1 = EvaluateRetrieval.precision_at_k(results, qrels, k=1)

        # Return the NDCG@2, recall@1, and precision@1.
        return ndcg2, recall1, precision1

if __name__ == "__main__":
    # Load the corpus and queries.
    data_path = "./datasets/nq"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # Create the retrieval model.
    base_name = "GPL/msmarco-distilbert-margin-mse"
    gpl_name = "qberg/gpl-23m-bio"
    model = DRES(models.SentenceBERT(gpl_name), batch_size=128)

    # Create the LightningModule.
    retriever = GPLRetrieval(model)

    # Create the Trainer.
    logger = wandb("logs")
    trainer = pl.Trainer(logger=logger)

    # Train the model.
    trainer.fit(retriever)

    # Evaluate the model.
    results = trainer.test(retriever)

    # Print the NDCG@2, recall@1, and precision@1.
    ndcg2 = results["test_ndcg2"]
    recall1 = results["test_recall1"]
    precision1 = results["test_precision1"]
    print(f"NDCG@2: {ndcg2:.4f}")
    print(f"Recall@1: {recall1:.4f}")
    print(f"Precision@1: {precision1:.4f}")
