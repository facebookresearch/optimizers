"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import warnings
from copy import deepcopy

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

###### ENUM CLASSES ######
class PreconditionerType(enum.IntEnum):
    FULL = 0
    DIAGONAL = 1


class GraftingType(enum.IntEnum):
    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4


class RootInvMethod(enum.IntEnum):
    EIGEN = 0
    NEWTON = 1


class LargeDimMethod(enum.IntEnum):
    DIAGONAL = 0
    ADAGRAD = 1
    BLOCKING = 2


class NewtonConvergenceFlag(enum.IntEnum):
    REACHED_MAX_ITERS = 0
    CONVERGED = 1


###### MERGING AND BLOCKING HELPER FUNCTIONS ######
def merge_small_dims(tensor_shape, threshold):
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_size (torch.tensor or np.array or List[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """
    if torch.is_tensor(tensor_shape):
        tensor_shape = tensor_shape.numpy()

    if len(tensor_shape) <= 1:
        return tensor_shape

    new_tensor_shape = [tensor_shape[0]]
    for i in range(1, len(tensor_shape)):
        new_dimension = new_tensor_shape[-1] * tensor_shape[i]
        if (
            new_tensor_shape[-1] == 1
            or tensor_shape[i] == 1
            or new_dimension <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(tensor_shape[i])

    return new_tensor_shape


def multi_dim_chunk(grad, num_splits):
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        grad (torch.tensor): Gradient or tensor to split.
        num_splits (List[int]): Number of splits/chunks.

    Returns:
        split_grad (List[torch.tensor]): List of tensors.

    """
    import torch

    split_grad = [grad]
    for dim, split in enumerate(num_splits):
        if split > 0:
            temp_grads = split_grad
            split_grad = []
            for grad in temp_grads:
                split_grad.extend(torch.chunk(grad, split, dim=dim))

    return split_grad


def multi_dim_cat(split_grad, num_splits):
    """Concatenates multiple tensors to form single tensor across multiple dimensions.

    Args:
        split_grad (List[torch.tensor]): List of gradient chunks.
        num_splits (List[int]): Number of splits/chunks.

    Returns:
        merged_grad (torch.tensor): Merged tensor.

    """
    merged_grad = split_grad
    for dim, split in reversed(list(enumerate(num_splits))):
        if split > 0:
            temp_grad = []
            for i in range(0, len(merged_grad), split):
                temp_grad.append(torch.cat(merged_grad[i : i + split], axis=dim))
            merged_grad = temp_grad

    return merged_grad[0]


###### FUNCTIONS FOR PRECONDITIONERS ######
class Preconditioner:
    """Preconditioner class for Shampoo algorithm."""

    def __init__(self):
        self.parameter_count = 0
        pass

    def update_preconditioners(self, grad):
        pass

    def precondition(self, grad):
        pass

    def precondition_and_update(self, param, grad, lr):
        pass

    def compute_norm(self, grad):
        pass

    def parameter_count(self):
        return self.parameter_count

    def broadcast(self, src_rank):
        return


class AdagradPreconditioner(Preconditioner):
    """Adagrad/Adam/RMSProp preconditioner for a generic layer.

    Stores preconditioner using same format as parameter p. Operations are performed in-place.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, bias_correction = True.

    Other variants can also be specified.

    Args:
        param (torch.tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-3)
        bias_correction (bool): Flag for using bias correction. (Default: False)
        idx (int or None): Layer index (for logging purposes). (Default: None)

    """

    def __init__(
        self,
        param,
        beta2=1.0,
        epsilon=1e-3,
        bias_correction=False,
        idx=None,
    ):
        super(AdagradPreconditioner, self).__init__()
        self.beta2 = beta2
        self.epsilon = epsilon
        self.preconditioner = torch.zeros_like(
            param, memory_format=torch.preserve_format
        )
        self.idx = idx
        self.num_updates = 0
        self.bias_correction = bias_correction
        self.bias_correction2 = 1.0
        self.parameter_count += torch.prod(torch.tensor(self.preconditioner.shape))

        if self.idx is not None:
            logger.info(f"Diagonal Adagrad Preconditioner with Parameter {self.idx}")

    def update_preconditioners(self, grad):
        if self.beta2 == 1.0:
            self.preconditioner.addcmul_(grad, grad, value=1)
        else:
            self.preconditioner.mul_(self.beta2).addcmul_(
                grad, grad.conj(), value=1 - self.beta2
            )

        self.num_updates += 1
        if self.bias_correction and self.beta2 < 1.0:
            self.bias_correction2 = 1.0 - self.beta2 ** self.num_updates

    def precondition(self, grad):
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        grad.div_(denom)
        return grad

    def precondition_and_update(self, param, grad, lr):
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        param.addcdiv_(grad, denom, value=-lr)

    def compute_norm(self, grad):
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        adagrad_nrm = torch.linalg.norm(grad / denom)
        return adagrad_nrm


class ShampooPreconditioner(Preconditioner):
    """Shampoo preconditioners for some generic layer.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (torch.tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update.
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness.
        bias_correction (bool): Flag for using bias correction.
        diagonal_threshold (int): Threshold for using diagonal preconditioners. If None, disabled.
        large_dim_method (LargeDimMethod): Specifies which large-dimensional method to use.
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners.
        root_inv_method (str): Specifies method for computing root inverse. Coupled inverse Newton iteration
            is only more efficient on GPU for dimensions <= 1024.
        idx (int or None): Layer index (for logging purposes).
        init_delay (int, optional): initial delay before starting to compute root inverse. Applies grafting method beforehand. (default: 0)
        grafting_type (GraftingType, optional): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float, optional): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float, optional): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2=1.0,
        epsilon=1e-12,
        bias_correction=False,
        diagonal_threshold=None,
        dtype=torch.float,
        root_inv_method=RootInvMethod.EIGEN,
        idx=None,
        init_delay=0,
        grafting_type=GraftingType.NONE,
        grafting_beta2=1.0,
        grafting_epsilon=1e-3,
    ):

        super(ShampooPreconditioner, self).__init__()

        # initialize parameters
        self.beta2 = beta2
        self.epsilon = epsilon
        self.diagonal_threshold = diagonal_threshold
        self.dtype = dtype
        self.num_updates = 0
        self.bias_correction = bias_correction
        self.bias_correction2 = 1.0
        self.dims = torch.tensor(param.shape).numpy()
        self.order = param.dim()
        self.idx = idx
        self.root_inv_method = root_inv_method
        self.grafting_type = grafting_type
        self.init_delay = init_delay

        # initialize lists for each parameter
        self.preconditioners = []
        self.inv_preconditioners = []
        self.preconditioner_types = []

        for k in range(self.order):
            dim = self.dims[k]

            if self.diagonal_threshold is not None and dim > self.diagonal_threshold:
                preconditioner = torch.zeros(
                    dim, dtype=param.dtype, device=param.device
                )
                inv_preconditioner = None
                preconditioner_type = PreconditionerType.DIAGONAL
                num_params = dim
                if self.idx is not None:
                    logger.info(
                        f"Diagonal Preconditioner with Parameter {self.idx}, Order {k}, Dim {dim}, Number of Params {num_params}"
                        + ", DType "
                        + str(self.dtype)
                    )
            else:
                preconditioner = torch.zeros(
                    (dim, dim), dtype=self.dtype, device=param.device
                )
                inv_preconditioner = torch.zeros(
                    (dim, dim), dtype=param.dtype, device=param.device
                )
                preconditioner_type = PreconditionerType.FULL
                num_params = dim ** 2
                if self.idx is not None:
                    logger.info(
                        f"Full Matrix Preconditioner with Parameter {self.idx}, Order {k}, Dim {dim}, Number of Params {num_params}"
                        + ", DType "
                        + str(self.dtype)
                    )

            self.parameter_count += num_params
            self.preconditioners.append(preconditioner)
            self.inv_preconditioners.append(inv_preconditioner)
            self.preconditioner_types.append(preconditioner_type)

        # initialize grafting
        if self.grafting_type == GraftingType.NONE:
            self.grafting = None
        elif self.grafting_type == GraftingType.SGD:
            self.grafting = SGDGrafting(param)
        elif self.grafting_type == GraftingType.ADAGRAD:
            self.grafting = AdagradGrafting(param, epsilon=grafting_epsilon)
        elif self.grafting_type == GraftingType.RMSPROP:
            self.grafting = RMSPropGrafting(
                param,
                epsilon=grafting_epsilon,
                beta2=grafting_beta2,
            )
        elif self.grafting_type == GraftingType.ADAM:
            self.grafting = AdamGrafting(
                param,
                epsilon=grafting_epsilon,
                beta2=grafting_beta2,
                bias_correction=True,
            )
        else:
            raise ValueError("Invalid Grafting Type!")

        if self.grafting_type != GraftingType.NONE:
            self.parameter_count += self.grafting.parameter_count

    def update_preconditioners(self, grad):

        # iterate over all dimensions
        for k in range(self.order):
            preconditioner = self.preconditioners[k]
            preconditioner_type = self.preconditioner_types[k]
            dim = self.dims[k]

            # update preconditioners (diagonal case)
            if preconditioner_type == PreconditionerType.DIAGONAL:

                # Adagrad accumulation
                if self.beta2 == 1.0:
                    preconditioner.add_(
                        torch.linalg.norm(
                            grad.transpose(0, k).contiguous().view(dim, -1),
                            dim=1,
                        ).pow(2)
                    )

                # Exponential moving average
                else:
                    preconditioner.mul_(self.beta2).add_(
                        torch.linalg.norm(
                            grad.transpose(0, k).contiguous().view(dim, -1),
                            dim=1,
                        ).pow(2),
                        alpha=1 - self.beta2,
                    )

            # update preconditioners (full-matrix case)
            else:

                # Adagrad accumulation
                if self.beta2 == 1.0:
                    contract_idx = [*range(k)] + [*range(k + 1, self.order)]
                    preconditioner.add_(
                        torch.tensordot(
                            grad,
                            grad,
                            dims=(contract_idx, contract_idx),
                        ).to(self.dtype)
                    )

                # Exponential moving average
                else:
                    contract_idx = [*range(k)] + [*range(k + 1, self.order)]
                    preconditioner.mul_(self.beta2).add_(
                        torch.tensordot(
                            grad,
                            grad,
                            dims=(contract_idx, contract_idx),
                        ).to(self.dtype),
                        alpha=1 - self.beta2,
                    )

        # update grafting method
        if self.grafting_type != GraftingType.NONE:
            self.grafting.update_preconditioners(grad)

        self.num_updates += 1
        if self.bias_correction and self.beta2 < 1.0:
            self.bias_correction2 = 1.0 - self.beta2 ** self.num_updates

    def precondition(self, grad):

        preconditioned_grad = grad.clone()

        # iterate over all dimensions
        for k in range(self.order):
            preconditioner = self.preconditioners[k]
            inv_preconditioner = self.inv_preconditioners[k]
            preconditioner_type = self.preconditioner_types[k]

            # handle diagonal case while retaining dims
            if self.diagonal_threshold is not None:

                # precondition in diagonal case
                if preconditioner_type == PreconditionerType.DIAGONAL:
                    denom = (preconditioner / self.bias_correction2).add_(self.epsilon)
                    preconditioned_grad.div_(
                        denom.pow(-1 / (2 * self.order))[
                            (None,) * k + (...,) + (None,) * (self.order - k - 1)
                        ]
                    )

                # precondition in full-matrix case
                else:
                    gradient_idx = [*range(1, self.order + 1)]
                    matrix_product_idx = deepcopy(gradient_idx)
                    matrix_product_idx[k] = 0
                    preconditioned_grad = torch.einsum(
                        inv_preconditioner,
                        [0, k + 1],
                        preconditioned_grad,
                        gradient_idx,
                        matrix_product_idx,
                    )

            # more efficient but transposes grad continually
            else:
                preconditioned_grad = torch.tensordot(
                    preconditioned_grad, inv_preconditioner, [[0], [0]]
                )

        # apply grafting
        if self.grafting_type != GraftingType.NONE:
            grafting_norm = self.grafting.direction_norm(grad)
            preconditioned_grad = (
                preconditioned_grad
                * grafting_norm
                / (torch.linalg.norm(preconditioned_grad) + 1e-16)
            )

        return preconditioned_grad

    def compute_root_inverse(self, debug=False):

        # array of residuals and cond_numbers
        residuals = None
        cond_numbers = None
        eigenvalue_gaps = None

        # iterate over all dimensions
        for k in range(self.order):
            preconditioner = self.preconditioners[k]
            preconditioner_type = self.preconditioner_types[k]

            # check that this is a full matrix preconditioner
            if preconditioner_type == PreconditionerType.FULL:
                # add epsilon term and incorporate bias correction
                bias_corrected_preconditioner = preconditioner / self.bias_correction2

                # check if nan or inf values
                if torch.any(torch.isnan(bias_corrected_preconditioner)):
                    logger.info(
                        f"Encountered nan values in preconditioner {self.idx}.{k}!"
                    )
                elif torch.any(torch.isinf(bias_corrected_preconditioner)):
                    logger.info(
                        f"Encountered inf values in preconditioner {self.idx}.{k}!"
                    )

                # compute inverse preconditioner and store
                root = 2 * self.order
                if self.root_inv_method == RootInvMethod.EIGEN:
                    (inv_preconditioner, eigvals, eigvecs,) = matrix_inverse_root_eigen(
                        A=bias_corrected_preconditioner,
                        root=root,
                        perturb=True,
                        epsilon=self.epsilon,
                    )
                    cond_number = (
                        torch.max(eigvals) / torch.min(eigvals) if debug else None
                    )
                    eigenvalue_gap = (
                        torch.min(eigvals[1:] - eigvals[:-1])
                        if debug and eigvals.shape[0] > 1
                        else None
                    )
                elif self.root_inv_method == RootInvMethod.NEWTON:
                    (
                        inv_preconditioner,
                        _,
                        flag,
                        num_iterations,
                        error,
                    ) = matrix_inverse_root_newton(
                        A=bias_corrected_preconditioner, root=root, epsilon=self.epsilon
                    )
                    cond_number = None
                    eigenvalue_gap = None
                    if flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
                        warnings.warn(
                            f"Reached maximum number of iterations {num_iterations} with error {error} in coupled inverse Newton iteration!"
                        )
                else:
                    raise ValueError("Invalid method. Only supports eigen and newton.")

                inv_preconditioner = inv_preconditioner.to(
                    dtype=self.inv_preconditioners[k].dtype
                )
                self.inv_preconditioners[k] = inv_preconditioner

                # compute preconditioner norm and residual
                if debug:
                    regularized_preconditioner = (
                        bias_corrected_preconditioner
                        + self.epsilon
                        * torch.eye(
                            preconditioner.shape[0],
                            dtype=preconditioner.dtype,
                            device=preconditioner.device,
                        )
                    )
                    preconditioner_norm = torch.linalg.norm(
                        regularized_preconditioner, ord=torch.inf
                    )
                    residual = (
                        torch.dist(
                            torch.matrix_power(inv_preconditioner, -root),
                            regularized_preconditioner,
                            p=torch.inf,
                        )
                        / torch.maximum(torch.tensor(1.0), preconditioner_norm)
                    )

                    # append residual and condition number to arrays
                    if residuals is None:
                        residuals = residual.cpu().unsqueeze(-1)
                    else:
                        residuals = torch.cat(
                            (residuals, residual.cpu().unsqueeze(-1)), 0
                        )
                    if self.root_inv_method == RootInvMethod.EIGEN:
                        if cond_numbers is None:
                            cond_numbers = cond_number.cpu().unsqueeze(-1)
                        else:
                            cond_numbers = torch.cat(
                                (cond_numbers, cond_number.cpu().unsqueeze(-1)), 0
                            )
                        if eigenvalue_gap is not None:
                            if eigenvalue_gaps is None:
                                eigenvalue_gaps = eigenvalue_gap.cpu().unsqueeze(-1)
                            else:
                                eigenvalue_gaps = torch.cat(
                                    (
                                        eigenvalue_gaps,
                                        eigenvalue_gap.cpu().unsqueeze(-1),
                                    ),
                                    0,
                                )

        return residuals, cond_numbers, eigenvalue_gaps

    def precondition_and_update(self, param, grad, lr):
        if self.num_updates <= self.init_delay:
            self.grafting.precondition_and_update(param, grad, lr)
        else:
            preconditioned_grad = self.precondition(grad)
            param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad):
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self, src_rank):
        for k in range(self.order):
            if self.preconditioner_types[k] == PreconditionerType.FULL:
                dist.broadcast(
                    self.inv_preconditioners[k],
                    src=src_rank,
                )


class BlockShampooPreconditioner(Preconditioner):
    """Shampoo with blocking applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (torch.tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update.
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness.
        bias_correction (bool): Flag for using bias correction.
        block_size (int): Block size for blocking large tensors.
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners.
        root_inv_method (str): Specifies method for computing root inverse. Coupled inverse Newton iteration
            is only more efficient on GPU for dimensions <= 1024.
        idx (int or None): Layer index (for logging purposes).
        merge_dims (bool): Denotes whether or not dimensions are merged.
        init_delay (int, optional): initial delay before starting to compute root inverse. Applies grafting method beforehand. (default: 0)
        grafting_type (LayerwiseGraftingType, optional): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float, optional): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float, optional): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2=1.0,
        epsilon=1e-3,
        bias_correction=False,
        block_size=1024,
        dtype=torch.double,
        root_inv_method=RootInvMethod.EIGEN,
        idx=None,
        merge_dims=True,
        init_delay=0,
        grafting_type=GraftingType.NONE,
        grafting_beta2=1.0,
        grafting_epsilon=1e-3,
    ):
        super(BlockShampooPreconditioner, self).__init__()

        # Set hyperparameters
        self.beta2 = beta2
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.block_size = block_size
        self.dtype = dtype
        self.num_updates = 0
        self.root_inv_method = root_inv_method
        self.idx = idx
        self.init_delay = init_delay
        self.merge_dims = merge_dims
        self.original_dims = [*torch.tensor(param.shape).numpy()]
        self.merged_dims = (
            merge_small_dims(self.original_dims, self.block_size)
            if self.block_size is not None and merge_dims
            else self.original_dims
        )
        self.original_order = param.dim()
        self.merged_order = len(self.merged_dims) if merge_dims else self.original_order

        # Construct splits for blocking
        self.splits = (
            torch.ceil(torch.tensor(self.merged_dims) / block_size)
            .to(dtype=torch.int)
            .numpy()
        )

        # Construct multiple preconditioners for each block
        self.split_preconditioners = []
        self.split_sizes = []

        if self.merge_dims:
            param = param.view(self.merged_dims)

        split_param = multi_dim_chunk(param, self.splits)
        for i, p in enumerate(split_param):
            self.split_sizes.append(torch.tensor(p.shape))
            split_idx = float(str(idx) + "." + str(i))
            preconditioner = ShampooPreconditioner(
                p,
                beta2=beta2,
                epsilon=epsilon,
                bias_correction=bias_correction,
                dtype=dtype,
                root_inv_method=root_inv_method,
                idx=split_idx,
                init_delay=init_delay,
                grafting_type=grafting_type,
                grafting_beta2=grafting_beta2,
                grafting_epsilon=grafting_epsilon,
            )
            self.split_preconditioners.append(preconditioner)
            self.parameter_count += preconditioner.parameter_count

    def update_preconditioners(self, grad):
        if self.merge_dims:
            grad = grad.view(self.merged_dims)
        split_grad = multi_dim_chunk(grad, self.splits)
        for i, g in enumerate(split_grad):
            self.split_preconditioners[i].update_preconditioners(g)
        self.num_updates += 1

    def precondition(self, grad):
        if self.merge_dims:
            grad = grad.view(self.merged_dims)
        split_grad = multi_dim_chunk(grad, self.splits)
        split_preconditioned_grad = []
        for i, g in enumerate(split_grad):
            if self.num_updates <= self.init_delay:
                preconditioned_g = self.split_preconditioners[i].grafting.precondition(
                    g
                )
            else:
                preconditioned_g = self.split_preconditioners[i].precondition(g)
            split_preconditioned_grad.append(preconditioned_g)
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self.splits)
        if self.merge_dims:
            preconditioned_grad = preconditioned_grad.view(self.original_dims)
        return preconditioned_grad

    def compute_root_inverse(self, debug=False):
        residuals = None
        cond_numbers = None
        eigenvalue_gaps = None
        for i in range(len(self.split_preconditioners)):
            (
                split_residuals,
                split_cond_numbers,
                split_eigenvalue_gaps,
            ) = self.split_preconditioners[i].compute_root_inverse(debug=debug)

            if residuals is None:
                residuals = split_residuals
            else:
                residuals = torch.cat((residuals, split_residuals), 0)
            if self.root_inv_method == RootInvMethod.EIGEN:
                if cond_numbers is None:
                    cond_numbers = split_cond_numbers
                else:
                    cond_numbers = torch.cat((cond_numbers, split_cond_numbers), 0)
                if eigenvalue_gaps is None:
                    eigenvalue_gaps = split_eigenvalue_gaps
                else:
                    eigenvalue_gaps = torch.cat(
                        (eigenvalue_gaps, split_eigenvalue_gaps), 0
                    )

        return residuals, cond_numbers, eigenvalue_gaps

    def precondition_and_update(self, param, grad, lr):
        preconditioned_grad = self.precondition(grad)
        param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad):
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self, src_rank):
        for i in range(len(self.split_preconditioners)):
            self.split_preconditioners[i].broadcast(src_rank)


###### FUNCTIONS FOR GRAFTING ######
class Grafting:
    def __init__(self, param):
        self.parameter_count = 0
        pass

    def update_preconditioners(self, grad):
        pass

    def precondition(self, grad):
        pass

    def direction_norm(self, grad):
        pass

    def precondition_and_update(self, param, grad, lr):
        pass


class SGDGrafting(Grafting):
    def __init__(self, param):
        super(SGDGrafting, self).__init__(param)

    def precondition(self, grad):
        return grad

    def direction_norm(self, grad):
        return torch.linalg.norm(grad)

    def precondition_and_update(self, param, grad, lr):
        param.add_(grad, alpha=-lr)


class AdagradGrafting(Grafting):
    def __init__(self, param, epsilon):
        super(AdagradGrafting, self).__init__(param)
        self.preconditioner = AdagradPreconditioner(param, epsilon=epsilon)
        self.parameter_count += self.preconditioner.parameter_count

    def update_preconditioners(self, grad):
        self.preconditioner.update_preconditioners(grad)

    def precondition(self, grad):
        return self.preconditioner.precondition(grad)

    def direction_norm(self, grad):
        return self.preconditioner.compute_norm(grad)

    def precondition_and_update(self, param, grad, lr):
        self.preconditioner.precondition_and_update(param, grad, lr)


class RMSPropGrafting(Grafting):
    def __init__(self, param, beta2, epsilon):
        super(RMSPropGrafting, self).__init__(param)
        self.preconditioner = AdagradPreconditioner(param, beta2=beta2, epsilon=epsilon)
        self.parameter_count += self.preconditioner.parameter_count

    def update_preconditioners(self, grad):
        self.preconditioner.update_preconditioners(grad)

    def precondition(self, grad):
        return self.preconditioner.precondition(grad)

    def direction_norm(self, grad):
        return self.preconditioner.compute_norm(grad)

    def precondition_and_update(self, param, grad, lr):
        self.preconditioner.precondition_and_update(param, grad, lr)


class AdamGrafting(Grafting):
    def __init__(self, param, beta2, epsilon):
        super(AdamGrafting, self).__init__(param)
        self.preconditioner = AdagradPreconditioner(
            param, beta2=beta2, epsilon=epsilon, bias_correction=True
        )
        self.parameter_count += self.preconditioner.parameter_count

    def update_preconditioners(self, grad):
        self.preconditioner.update_preconditioners(grad)

    def precondition(self, grad):
        return self.preconditioner.precondition(grad)

    def direction_norm(self, grad):
        return self.preconditioner.compute_norm(grad)

    def precondition_and_update(self, param, grad, lr):
        self.preconditioner.precondition_and_update(param, grad, lr)


###### FUNCTIONS FOR COMPUTING MATRIX ROOT INVERSE ######
def matrix_inverse_root_eigen(
    A, root: int, perturb: bool = False, inverse: bool = True, epsilon: float = 0.0
):
    """Compute matrix inverse root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A = Q L Q^T => A^{1/r} = Q L^{1/r} Q^T OR A^{-1/r} = Q L^{-1/r} Q^T

    Args:
        A (torch.tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        perturb (bool): Perturbs matrix eigenvalues to ensure it is (practically) positive semi-definite.
        inverse (bool): Returns inverse root matrix.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.

    Returns:
        X (torch.tensor): (Inverse) root of matrix. Same dimensions as A.
        L (torch.tensor): Eigenvalues of A.
        Q (torch.tensor): Orthogonal matrix consisting of eigenvectors of A.

    """

    # check if root is positive integer
    if root <= 0:
        raise ValueError("Root is not positive!")

    # compute matrix power
    alpha = 1 / root
    if inverse:
        alpha = -alpha

    # check if matrix is scalar
    if len(A.shape) == 0 or (len(A.shape) == 1 and A.shape[0] == 1):
        return A ** alpha
    elif len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    # compute eigendecomposition and compute maximum and minimum eigenvalues
    L, Q = torch.linalg.eigh(A)
    lambda_min = torch.min(L)
    lambda_max = torch.max(L)

    # perturb eigenvalues (if necessary)
    if perturb:
        L += -torch.minimum(lambda_min, torch.tensor(0.0))
        lambda_min += -torch.minimum(lambda_min, torch.tensor(0.0))
        lambda_max += -torch.minimum(lambda_min, torch.tensor(0.0))

    # add epsilon
    L += epsilon
    lambda_min += epsilon
    lambda_max += epsilon

    # compute inverse preconditioner
    X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

    return X, L, Q


def matrix_inverse_root_newton(
    A,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
):
    """Compute matrix square root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    Works faster than the eigendecomposition approach on GPU for dimensions smaller than 1024, but not for CPU.
    This is likely due to efficient matrix multiplication on GPU.

    Args:
        A (torch.tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root.

    Returns:
        A_root (torch.tensor): Inverse square root of matrix.
        M (torch.tensor): Coupled matrix.
        flag (int): Specifies convergence.
        iteration (int): Number of iterations.
        error (float): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A = A + epsilon * identity

    # initialize matrices
    A_nrm = torch.linalg.norm(A)
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A
    error = torch.dist(M, identity, p=torch.inf)

    # main for loop
    while error > tolerance and iteration < max_iterations:
        iteration += 1
        M_p = (1 - alpha) * identity + alpha * M
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = torch.max(torch.abs(M - identity))

    # determine convergence flag
    if error <= tolerance:
        flag = NewtonConvergenceFlag.CONVERGED
    else:
        flag = NewtonConvergenceFlag.REACHED_MAX_ITERS

    return X, M, flag, iteration, error
