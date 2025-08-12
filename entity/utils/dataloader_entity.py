from ..base_entity import BaseEntity


class DataloaderEntity(BaseEntity):
    bathc_size: int = 32
    num_worker: int = 0
    shuffle: bool = True