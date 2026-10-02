# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.distributed.device_mesh",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="nemo_automodel.components.moe.state_dict_utils",
)
