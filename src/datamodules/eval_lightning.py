import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import wandb

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

@hydra.main(config_path="example.yaml")
class GPLRetrieval(pl.LightningModule):
    """GPL retrieval model.

    Args:
        cfg (DictConfig): Configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        self.model = DRES(models.SentenceBERT(self.cfg.model.gpl_name), batch_size=self.cfg.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
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
    hydra.run(GPLRetrieval)
