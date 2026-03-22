from torch.utils.data import DataLoader
from dataset import DocRankingDataset


class DocRankingDataLoader(DataLoader):
    def __init__(self,
                 dataset: DocRankingDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 persistent_workers: bool = False) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate,
            # pin_memory=True pre-pages CPU tensors into pinned (non-pageable) memory,
            # which significantly speeds up CPU→GPU transfers on CUDA devices.
            pin_memory=pin_memory,
            # persistent_workers keeps worker processes alive between epochs,
            # avoiding the per-epoch fork/join overhead when num_workers > 0.
            persistent_workers=persistent_workers and num_workers > 0,
        )