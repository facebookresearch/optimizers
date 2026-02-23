"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Constants for optimizers - parameter group and state keys.
This is one single place to define all the keys used in optimizer state.
"""

from enum import Enum

# Generic keys
PARAMS = "params"

# Parameter group keys
LR = "lr"
EPS = "eps"
TRAIN_INTERP_COEFF = "train_interp_coeff"
BETA1 = "beta1"
BETA2 = "beta2"
WEIGHT_POW_COEFF = "weight_pow_coeff"
TRAIN_MODE = "train_mode"
LR_MAX = "lr_max"
WEIGHT_LR_POWER = "weight_lr_power"
WEIGHT_DECAY = "weight_decay"
EVAL_INTERP_COEFF = "eval_interp_coeff"
ITERATE_AVERAGING_TYPE = "iterate_averaging_type"

# Parameter state keys
Z_BUFFER = "z"
EXP_AVG = "exp_avg"
EXP_AVG_SQ = "exp_avg_sq"

# Stored in first parameter's state for shared access to avoid sync issues
STEP = "step"
WEIGHT_SUM = "weight_sum"


class IterateAveragingType(Enum):
    """Type of iterate averaging used in GPA optimizers.

    GPA: Generalized Primal Averaging with fixed eval_interp_coeff (mu_x).
    SCHEDULE_FREE: Schedule-Free averaging with polynomial weighting (no fixed mu_x).
    """

    GPA = "gpa"
    SCHEDULE_FREE = "schedule_free"
