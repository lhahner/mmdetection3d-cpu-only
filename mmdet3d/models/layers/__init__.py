# Copyright (c) OpenMMLab. All rights reserved.
from .._import_utils import optional_import

__all__ = []

optional_import('.box3d_nms',
                ['aligned_3d_nms', 'box3d_multiclass_nms', 'circle_nms',
                 'nms_bev', 'nms_normal_bev'], globals(), __all__)
optional_import('.dgcnn_modules',
                ['DGCNNFAModule', 'DGCNNFPModule', 'DGCNNGFModule'],
                globals(), __all__)
optional_import('.edge_fusion_module', ['EdgeFusionModule'], globals(), __all__)
optional_import('.fusion_layers',
                ['PointFusion', 'VoteFusion', 'apply_3d_transformation',
                 'bbox_2d_transform', 'coord_2d_transform'], globals(), __all__)
optional_import('.minkowski_engine_block',
                ['MinkowskiBasicBlock', 'MinkowskiBottleneck',
                 'MinkowskiConvModule'], globals(), __all__)
optional_import('.mlp', ['MLP'], globals(), __all__)
optional_import('.norm', ['NaiveSyncBatchNorm1d', 'NaiveSyncBatchNorm2d'],
                globals(), __all__)
optional_import('.paconv', ['PAConv', 'PAConvCUDA'], globals(), __all__)
optional_import('.pointnet_modules',
                ['PAConvCUDASAModule', 'PAConvCUDASAModuleMSG',
                 'PAConvSAModule', 'PAConvSAModuleMSG', 'PointFPModule',
                 'PointSAModule', 'PointSAModuleMSG', 'build_sa_module'],
                globals(), __all__)
optional_import('.sparse_block',
                ['SparseBasicBlock', 'SparseBottleneck',
                 'make_sparse_convmodule'], globals(), __all__)
optional_import('.torchsparse_block',
                ['TorchSparseBasicBlock', 'TorchSparseBottleneck',
                 'TorchSparseConvModule'], globals(), __all__)
optional_import('.transformer', ['GroupFree3DMHA'], globals(), __all__)
optional_import('.vote_module', ['VoteModule'], globals(), __all__)
