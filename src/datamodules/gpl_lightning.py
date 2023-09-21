import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb

from gpl import GPLModel

class GPLLightningModule(pl.LightningModule):
    def __init__(
        self,
        path_to_generated_data: str,
        base_ckpt: str,
        batch_size_gpl: int,
        gpl_steps: int,
        output_dir: str,
        generator: str,
        retrievers: list,
        cross_encoder: str,
        qgen_prefix: str,
        do_evaluation=False,
    ):
        super().__init__()

        self.path_to_generated_data = path_to_generated_data
        self.base_ckpt = base_ckpt
        self.batch_size_gpl = batch_size_gpl
        self.gpl_steps = gpl_steps
        self.output_dir = output_dir
        self.generator = generator
        self.retrievers = retrievers
        self.cross_encoder = cross_encoder
        self.qgen_prefix = qgen_prefix
        self.do_evaluation = do_evaluation

        self.gpl_model = GPLModel(
            path_to_generated_data,
            base_ckpt,
            batch_size_gpl,
            gpl_steps,
            output_dir,
            generator,
            retrievers,
            cross_encoder,
            qgen_prefix,
            do_evaluation,
        )

    def configure_optimizers(self):
        return self.gpl_model.optimizer

    def training_step(self, batch, batch_idx):
        # Train the GPL model.
        loss = self.gpl_model.train_step(batch)

        return loss

    def save_model(self):
        self.gpl_model.save_model(self.output_dir)

if __name__ == "__main__":
    # Create the Lightning module.
    model = GPLLightningModule(
        path_to_generated_data='generated/trec-covid',
        base_ckpt='qberg/gpl-23m-bio',
        batch_size_gpl=32,
        gpl_steps=140_000,
        output_dir='./output/custom_model',
        generator='BeIR/query-gen-msmarco-t5-base-v1',
        retrievers=[
            'msmarco-distilbert-base-v3',
            'msmarco-MiniLM-L-6-v3'
        ],
        cross_encoder='cross-encoder/ms-marco-MiniLM-L-6-v2',
        qgen_prefix='qgen',
        do_evaluation=False
    )

    # Create the Trainer.
    logger = wandb("logs")
    trainer = pl.Trainer(logger=logger)

    # Train the model.
    trainer.fit(model)

    # Save the model.
    model.save_model()
