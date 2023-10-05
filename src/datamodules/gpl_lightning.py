import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import wandb
import pytorch_lightning as pl
from gpl import GPLModel

@hydra.main(config_path="example.yaml")
class GPLLightningModule(pl.LightningModule):
    """GPLLightningModule.

    Args:
        cfg (DictConfig): Configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Load the configuration from the YAML file.
        self.cfg = cfg

        # Create the GPL model.
        self.gpl_model = GPLModel(
            path_to_generated_data=self.cfg.gpl.path_to_generated_data,
            base_ckpt=self.cfg.gpl.base_ckpt,
            batch_size_gpl=self.cfg.gpl.batch_size_gpl,
            gpl_steps=self.cfg.gpl.gpl_steps,
            output_dir=self.cfg.gpl.output_dir,
            generator=self.cfg.gpl.generator,
            retrievers=self.cfg.gpl.retrievers,
            cross_encoder=self.cfg.gpl.cross_encoder,
            qgen_prefix=self.cfg.gpl.qgen_prefix,
            do_evaluation=self.cfg.gpl.do_evaluation,
        )

    def configure_optimizers(self):
        """Configure the optimizers for the model."""

        # Return the optimizer from the GPL model.
        return self.gpl_model.optimizer

    def training_step(self, batch, batch_idx):
        """Perform a training step."""

        # Train the GPL model.
        loss = self.gpl_model.train_step(batch)

        # Return the loss.
        return loss

    def save_model(self):
        """Save the model."""

        # Save the GPL model to the output directory.
        self.gpl_model.save_model(self.cfg.gpl.output_dir)

if __name__ == "__main__":
    hydra.run(GPLLightningModule)
