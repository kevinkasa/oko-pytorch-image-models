import pdb

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        """
        Perform an all_gather but keep track of which rank we are on
        so we can handle gradients properly.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Prepare a list for gathering
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)

        # Save info for backward
        ctx.world_size = world_size
        ctx.rank = rank
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        On backward, we have a gradient for each rank's output in grad_outputs.
        We reduce these gradients back to each rank so that each process gets the
        gradient corresponding to its original input only.
        """
        world_size = ctx.world_size
        rank = ctx.rank

        grad_list = list(grad_outputs)  # tuple -> list

        # Asynchronously reduce gradients to the owners
        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(world_size)
        ]

        for op in dist_ops:
            op.wait()

        # After reduce, each grad_list[i] is the sum of gradients from all ranks
        # Only the gradient at position `rank` belongs to this rank's input
        # The rest are not needed by this rank
        return grad_list[rank]


def all_gather_with_grad(tensor):
    return AllGatherWithGrad.apply(tensor)


class OkoSetLoss(nn.Module):
    """
    Custom loss that:
    1. Gathers all logits and labels across distributed GPUs.
    2. Forms triplet sets (anchor, positive, negative) from the entire global batch.
    3. For each valid set, sums the logits of the three samples.
    4. Applies standard cross-entropy on these summed logits.
    5. Only computes loss for sets anchored on the current GPU.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Logits of shape [local_batch_size, num_classes].
            target (torch.Tensor): Integer class labels of shape [local_batch_size].

        Returns:
            torch.Tensor: Scalar loss.
        """
        device = x.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Gather logits and targets from all GPUs if in distributed mode
        if world_size > 1:
            # Use the custom all_gather with gradient function for logits
            gathered_x = all_gather_with_grad(x)  # returns a tuple of length world_size
            gathered_x = list(gathered_x)

            # Replace the current GPU's portion with the original x that has grad_fn
            gathered_x[rank] = x

            # Concatenate along the batch dimension
            all_x = torch.cat(gathered_x, dim=0)

            # Gather targets using standard all_gather (no grad needed for targets)
            all_target_list = [torch.zeros_like(target) for _ in range(world_size)]
            dist.all_gather(all_target_list, target)
            all_target = torch.cat(all_target_list, dim=0)
        else:
            # Not distributed
            all_x = x
            all_target = target

        global_batch_size = all_x.size(0)
        # num_classes = all_x.size(1) # Not strictly needed

        # Build a mapping from label to indices
        label_to_indices: dict[int: list] = {}  # {label: [idx]}
        all_targets_np = all_target.cpu().tolist()
        for idx, lbl in enumerate(all_targets_np):
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)

        # Determine which samples belong to this local rank
        local_batch_size = x.size(0)
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        # Attempt to form sets for each sample in the global batch
        for anchor_idx in range(global_batch_size):
            anchor_label = all_targets_np[anchor_idx]
            # Positive samples: others with the same label
            positive_pool = [i for i in label_to_indices[anchor_label] if i != anchor_idx]
            if len(positive_pool) == 0:
                # Cannot form a set for this anchor
                continue
            positive_idx = random.choice(positive_pool)

            # Negative samples: samples with a different label
            negative_pool = []
            for lbl, idxs in label_to_indices.items():
                if lbl != anchor_label:
                    negative_pool.extend(idxs)
            if len(negative_pool) == 0:
                # No negatives found
                continue
            negative_idx = random.choice(negative_pool)

            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)

        # Filter to keep only sets where the anchor belongs to the current GPU
        local_mask = [(start_idx <= a < end_idx) for a in anchor_indices]
        print(local_mask)
        if not any(local_mask):
            # No valid sets for this GPU
            return torch.tensor(0.0, device=device, requires_grad=True)

        anchor_indices = [a for a, m in zip(anchor_indices, local_mask) if m]
        positive_indices = [p for p, m in zip(positive_indices, local_mask) if m]
        negative_indices = [n for n, m in zip(negative_indices, local_mask) if m]

        # Convert indices to tensors
        anchor_indices_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
        positive_indices_t = torch.tensor(positive_indices, dtype=torch.long, device=device)
        negative_indices_t = torch.tensor(negative_indices, dtype=torch.long, device=device)

        # Gather the corresponding logits
        anchor_logits = all_x[anchor_indices_t]
        positive_logits = all_x[positive_indices_t]
        negative_logits = all_x[negative_indices_t]

        # Sum the logits of anchor, positive, and negative samples
        summed_logits = anchor_logits + positive_logits + negative_logits

        # Targets for these sets are the anchor's target labels
        target_indices = all_target[anchor_indices_t]

        # Apply standard cross-entropy on the summed logits
        loss = F.cross_entropy(summed_logits, target_indices)

        return loss
