from torchvision.transforms import Compose, Normalize, ToTensor

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from util import prepare_model


# [1] Read the training data (in Parquet file format).
path = "s3://air-example-data/data-cuj/train"
ds = ray.data.read_parquet(path)

# [2] Preprocess the training data.
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

def transform_image(row):
    row["image"] = transform(row["image"])
    return row

ds = ds.map(transform_image)

# [3] Ingest the training data to model in each training worker.
def train_func_per_worker():
    # Prepare model.
    from tqdm import tqdm
    model, loss_fn, optimizer = prepare_model()

    # Model training loop for 2 epochs.
    for epoch in range(2):
        train_data_iterator = ray.train.get_dataset_shard("train")
        train_data_iterator = train_data_iterator.iter_torch_batches(batch_size=32)

        model.train()
        for batch in tqdm(train_data_iterator, desc=f"Train Epoch {epoch}"):
            images = batch["image"]
            labels = batch["label"]
            pred = model(images)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

# [4] Start distributed training.
if __name__ == "__main__":
    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        datasets={"train": ds}
    )

    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")