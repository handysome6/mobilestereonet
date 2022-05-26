from .dataset import MiddleBuryDataset, SceneFlowDataset, KITTIDataset, DrivingStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "middlebury": MiddleBuryDataset,
}
