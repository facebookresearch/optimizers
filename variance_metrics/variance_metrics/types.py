# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import field


@dataclasses.dataclass
class MeanMetrics:
    r"""
    Dataclass for storing mean gradient metrics.

    Attributes:
        mean_l2_sq (float): L2-norm sq (euclidean norm sq) of the mean gradient
        mean_l1 (float): L1-norm of the mean gradient
        mean_linf (float): L-infinity norm of the mean gradient

    Dataclass operations:
        __add__ : Returns the combined norm metrics as if the two mean gradients were concatenated.
            For L2-squared and L1 norms, this is the sum of the respective metrics:
                \|[g1, g2]_2\|^2 = \|g1\|_2^2 + \|g2\|_2^2 and
                \|[g1, g2]_1\| = \|g1\|_1 + \|g2\|_1
            For the L-infinity norm, this is the maximum of the two norms:
                \|[g1, g2]_\infty\| = \max(\|g1\|_\infty, \|g2\|_\infty)
    """

    mean_l2_sq: float = 0.0
    mean_l1: float = 0.0
    mean_linf: float = 0.0
    nuc_norm: float = 0.0
    """Nuclear norm of the mean gradient (only populated under the spectral
    variance path). Sum-additive across independent parameter groups."""

    def __add__(self, other):
        if not isinstance(other, MeanMetrics):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
        return MeanMetrics(
            mean_l2_sq=self.mean_l2_sq + other.mean_l2_sq,
            mean_l1=self.mean_l1 + other.mean_l1,
            mean_linf=max(self.mean_linf, other.mean_linf),
            nuc_norm=self.nuc_norm + other.nuc_norm,
        )


@dataclasses.dataclass
class VarianceMetrics:
    r"""
    Dataclass for storing variance metrics.

    Attributes:
        variance (float): Variance or mean-squared error based on the euclidean norm, i.e.,
            MSE(g) = E[\|g - \nabla F\|_2^2] = \sum_{i = 1}^n \sigma(g_i)^2
        std_l1 (float): L1-norm (or sum) of the standard deviations for different components, i.e.,
            std_l1(g) = \sum_{i = 1}^n \sigma(g_i)
        std_inf (float): L-infinity norm or max of the standard deviations of different components, i.e.,
            std_max(g) = \max_{i = 1, ..., n} \sigma(g_i)

    Dataclass operations:
        All operations are based on the asumptions that all random variables are independent.

        __add__ : Adds variance metrics as the combined traces of the covariance of the concatenated random variables.
            Mathematically, for random vectors g_1 and g_2 with covariances Σ_1 and Σ_2:
                variance([g1, g2]) = variance(g1) + variance(g2)
                std_l1([g1, g2]) = std_l1(g1) + std_l1(g2)
                std_linf([g1, g2]) = max(std_linf(g1), std_linf(g2))

        __mul__ : Scales the variance metrics as if the number of samples used to estimate
            the mean of a random variable has been divided by the given scalar (not by scaling the random variables themselves).
            Thus, if g1 is the mean of n samples, VarianceMetrics(g1)*2 = VarianceMetrics(g2) results in metrics for g2,
            a random variable estimating the mean with n/2 samples.
                variance(g2) = variance(g1) * 2
                std_l1(g2) = std_l1(g1) * \sqrt{2}
                std_linf(g2) = std_linf(g1) * \sqrt{2}

        __truediv__  : Scales the variance metrics as if the number of samples used to estimate
            the mean of a random variable has been multiplied by the given scalar (not by scaling the random variables themselves).
            Thus, if g1 is the mean of n samples, VarianceMetrics(g1) / 2 = VarianceMetrics(g2) results in metrics for g2,
            a random variable estimating the mean with n*2 samples.
                variance(g2) = variance(g1) / 2
                std_l1(g2) = std_l1(g1) / \sqrt{2}
                std_linf(g2) = std_linf(g1) / \sqrt{2}
    """

    variance: float = 0.0
    std_l1: float = 0.0
    std_linf: float = 0.0
    nuclear_norm: float = 0.0
    """Nuclear norm of the std-deviation matrix (sum of sqrts of the
    eigenvalues of the sample-covariance gram matrix). Only populated under
    the spectral variance path."""

    def __add__(self, other):
        if not isinstance(other, VarianceMetrics):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
        return VarianceMetrics(
            variance=self.variance + other.variance,
            std_l1=self.std_l1 + other.std_l1,
            std_linf=max(self.std_linf, other.std_linf),
            nuclear_norm=self.nuclear_norm + other.nuclear_norm,
        )

    def __mul__(self, scale: int | float):
        if not isinstance(scale, (int, float)):
            raise TypeError(f"Cannot multiply {type(self)} and {type(scale)}")
        if scale < 0:
            raise ValueError(f"Cannot scale {type(self)} with negative numbers")
        return VarianceMetrics(
            variance=self.variance * scale,
            std_l1=self.std_l1 * math.sqrt(scale),
            std_linf=self.std_linf * math.sqrt(scale),
            nuclear_norm=self.nuclear_norm * math.sqrt(scale),
        )

    def __truediv__(self, scale: int | float):
        if not isinstance(scale, (int, float)):
            raise TypeError(f"Cannot divide {type(self)} and {type(scale)}")
        if scale == 0:
            raise ZeroDivisionError(f"Cannot divide by zero with {scale=}!")
        if scale < 0:
            raise ValueError(f"Cannot scale {type(self)} with negative numbers")
        return VarianceMetrics(
            variance=self.variance / scale,
            std_l1=self.std_l1 / math.sqrt(scale),
            std_linf=self.std_linf / math.sqrt(scale),
            nuclear_norm=self.nuclear_norm / math.sqrt(scale),
        )


@dataclasses.dataclass
class AggregatedVarianceMetrics:
    """
    Dataclass for aggregating multiple sets of variance metrics.

    Attributes:
        variance_metrics_sample (VarianceMetrics): Variance metrics computed for the sample gradient, representing local batch gradients.
        variance_metrics_example (VarianceMetrics): Variance metrics scaled to reflect each individual example or datapoint (e.g., each sequence).
        variance_metrics_global (VarianceMetrics): Variance metrics scaled for the global gradient, representing the entire batch or dataset.

    Dataclass operations:
        __add__: Combines the aggregated variance metrics from two instances as if their
            corresponding random variables (e.g., gradients) were concatenated.
    """

    variance_metrics_sample: VarianceMetrics = field(default_factory=VarianceMetrics)
    variance_metrics_example: VarianceMetrics = field(default_factory=VarianceMetrics)
    variance_metrics_global: VarianceMetrics = field(default_factory=VarianceMetrics)

    def __add__(self, other):
        if not isinstance(other, AggregatedVarianceMetrics):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
        result = AggregatedVarianceMetrics(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in dataclasses.fields(AggregatedVarianceMetrics)
            }
        )

        return result
