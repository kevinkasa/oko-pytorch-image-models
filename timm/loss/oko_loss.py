import pdb

import random
from collections import defaultdict, deque

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


class MemoryBank:
    """A memory bank for storing embeddings and labels to supplement
    current batch samples when forming triplet sets.
    """

    def __init__(self, max_per_class=100):
        """
        Args:
            max_per_class (int): Maximum number of embeddings to store per class.
        """
        self.max_per_class = max_per_class
        # For each class, we store a FIFO queue of embeddings
        self.class_to_embeddings = defaultdict(lambda: deque(maxlen=max_per_class))

    def add_samples(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Add a batch of samples to the memory bank.

        Args:
            embeddings (torch.Tensor): Shape [N, feature_dim]
            labels (torch.Tensor): Shape [N]
        """
        # Move to CPU for long term storage if desired (optional optimization)
        # Here we keep them on GPU for simplicity
        embeddings = embeddings.detach()
        labels = labels.detach()

        # Add each sample to the corresponding class queue
        for emb, lbl in zip(embeddings, labels):
            self.class_to_embeddings[lbl.item()].append(emb)

    def get_positive(self, class_label: int) -> torch.Tensor:
        """Get a positive embedding for the given class label from the memory bank.

        Args:
            class_label (int): The class we need a positive sample for.

        Returns:
            torch.Tensor or None: An embedding of shape [feature_dim] if available, else None.
        """
        if class_label in self.class_to_embeddings and len(self.class_to_embeddings[class_label]) > 0:
            # # Return a recent positive sample (for variety, you could sample randomly)
            return self.class_to_embeddings[class_label][0]
            # pop() removes and returns the rightmost (most recent) element
            # return self.class_to_embeddings[class_label].pop()
        return None


class OkoSetLossHardK(nn.Module):
    """
    Custom loss that:
    1. Gathers all logits and labels across distributed GPUs.
    2. Forms triplet sets (anchor, positive, negative) from the entire global batch.
       - If no positive is found in the current batch for the anchor class,
         retrieves a positive from the memory bank.
    3. Selects the hardest negative from the current batch based on the anchor's label.
       Hardness = the probability assigned by the negative’s logits to the anchor’s class.
    4. Sums the logits of the anchor, chosen positive, and hardest negative.
    5. Applies standard cross-entropy on these summed logits.
    6. Only computes loss for sets anchored on the current GPU.
    7. Updates the memory bank with current batch’s logits and labels after computing loss.
    """

    def __init__(self, memory_bank: MemoryBank):
        super().__init__()
        self.memory_bank = memory_bank

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        mb_positives = []  # Will store (set_idx, mb_logit, lbl) for memory bank positives if used
        pdb.set_trace()
        # For each sample in the global batch, attempt to form sets
        for anchor_idx in range(global_batch_size):
            anchor_label = all_targets_np[anchor_idx]

            # Positive pool: others with the same label in the batch
            positive_pool = [i for i in label_to_indices[anchor_label] if i != anchor_idx]

            # Negative pool: samples with different label in the batch
            negative_pool = []
            for lbl, idxs in label_to_indices.items():
                if lbl != anchor_label:
                    negative_pool.extend(idxs)

            if len(negative_pool) == 0:
                # No negatives found at all, skip this anchor
                continue

            # Select the positive
            if len(positive_pool) == 0:
                # No positive in batch, try memory bank
                mb_pos = self.memory_bank.get_positive(anchor_label)
                if mb_pos is None:
                    # Still no positive found, skip
                    continue
                # Mark positive as coming from memory bank
                anchor_indices.append(anchor_idx)
                positive_indices.append(-1)  # Indicate memory bank positive
                # We will insert it after filtering
            else:
                # Choose a random positive from the batch
                positive_idx = random.choice(positive_pool)
                anchor_indices.append(anchor_idx)
                positive_indices.append(positive_idx)
            pdb.set_trace()

            # Hard negative mining from the current batch
            # Compute hardness: For each candidate in negative_pool, pick the one
            # that gives the highest probability to anchor_label.
            negative_logits_candidates = all_x[negative_pool]  # [num_candidates, num_classes]
            # Compute softmax probabilities
            probs = F.softmax(negative_logits_candidates, dim=-1)
            # Hardness = probability of anchor_label
            hardness_scores = probs[:, anchor_label]
            hardest_idx = torch.argmin(hardness_scores).item()
            hardest_negative = negative_pool[hardest_idx]

            # Add the chosen negative
            negative_indices.append(hardest_negative)

            # If we used a memory bank positive, record it
            if positive_indices[-1] == -1:
                # Store the memory bank positive info
                mb_positives.append((len(anchor_indices) - 1, mb_pos, anchor_label))

        # Filter to keep only sets where the anchor belongs to the current GPU
        local_mask = [(start_idx <= a < end_idx) for a in anchor_indices]
        if not any(local_mask):
            # No valid sets for this GPU
            # Update memory bank with current batch
            # self.memory_bank.add_samples(x, target)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Apply the local_mask filtering
        anchor_indices = [a for a, m in zip(anchor_indices, local_mask) if m]
        positive_indices = [p for p, m in zip(positive_indices, local_mask) if m]
        negative_indices = [n for n, m in zip(negative_indices, local_mask) if m]
        print(f'num sets: {len(anchor_indices)}')

        # Re-map memory bank positives after filtering
        old_to_new = {}
        new_set_idx = 0
        for i, m in enumerate(local_mask):
            if m:
                old_to_new[i] = new_set_idx
                new_set_idx += 1

        new_mb_positives = []
        for (old_set_idx, emb, lbl) in mb_positives:
            if old_set_idx in old_to_new:
                new_mb_positives.append((old_to_new[old_set_idx], emb, lbl))

        # Convert indices to tensors
        anchor_indices_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
        batch_positive_indices = [p for p in positive_indices if p >= 0]
        batch_positive_indices_t = (torch.tensor(batch_positive_indices, dtype=torch.long, device=device)
                                    if batch_positive_indices else None)
        negative_indices_t = torch.tensor(negative_indices, dtype=torch.long, device=device)

        # Gather corresponding logits
        anchor_logits = all_x[anchor_indices_t]
        negative_logits = all_x[negative_indices_t]

        # Handle positives:
        # Batch positives
        if batch_positive_indices_t is not None:
            batch_positive_logits = all_x[batch_positive_indices_t]
        else:
            batch_positive_logits = None

        # Memory bank positives
        final_positive_logits = []
        mb_pos_idx = 0
        batch_pos_idx = 0
        for p in positive_indices:
            if p >= 0:
                final_positive_logits.append(batch_positive_logits[batch_pos_idx])
                batch_pos_idx += 1
            else:
                # from memory bank
                mb_logit = new_mb_positives[mb_pos_idx][1]
                mb_pos_idx += 1
                final_positive_logits.append(mb_logit.to(device))
        final_positive_logits = torch.stack(final_positive_logits, dim=0)

        # Sum the logits: anchor, positive, negative
        summed_logits = anchor_logits + final_positive_logits + negative_logits

        # Targets for these sets are the anchor's target labels
        target_indices = all_target[anchor_indices_t]

        # Compute cross-entropy loss
        loss = F.cross_entropy(summed_logits, target_indices)

        # Update the memory bank with this batch's logits and labels
        self.memory_bank.add_samples(x, target)

        return loss


class OkoSetLoss(nn.Module):
    """
    Custom loss that:
    1. Gathers all logits and labels across distributed GPUs.
    2. Forms triplet sets (anchor, positive, negative) from the entire global batch.
    3. For each valid set, sums the logits of the three samples.
    4. Applies standard cross-entropy on these summed logits.
    5. Only computes loss for sets anchored on the current GPU.
    """

    def __init__(self, memory_bank):
        super().__init__()
        self.memory_bank = memory_bank

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
        # memory_bank_positives = []  # Will store embeddings if we use memory bank for positives
        mb_positives = []  # To store memory bank positives (logits) if used

        # Attempt to form sets for each sample in the global batch
        for anchor_idx in range(global_batch_size):
            anchor_label = all_targets_np[anchor_idx]
            # Positive samples: others with the same label
            positive_pool = [i for i in label_to_indices[anchor_label] if i != anchor_idx]
            # if len(positive_pool) == 0:
            #     # Cannot form a set for this anchor
            #     continue
            # positive_idx = random.choice(positive_pool)

            # Negative samples: samples with a different label
            negative_pool = []
            for lbl, idxs in label_to_indices.items():
                if lbl != anchor_label:
                    negative_pool.extend(idxs)
            if len(negative_pool) == 0:
                # No negatives found
                continue
            negative_idx = random.choice(negative_pool)
            if len(positive_pool) == 0:
                # Try the memory bank if no positive in current batch
                mb_pos = self.memory_bank.get_positive(anchor_label)
                if mb_pos is None:
                    # No positive in memory bank, skip this anchor
                    continue
                # We'll store a placeholder for positive index
                # Because we do not have a batch positive index
                # We'll store (set_idx, mb_pos, anchor_label) to handle after filtering
                anchor_indices.append(anchor_idx)
                positive_indices.append(-1)  # indicate memory bank positive
                negative_indices.append(negative_idx)
                mb_positives.append((len(anchor_indices) - 1, mb_pos, anchor_label))
            else:
                positive_idx = random.choice(positive_pool)
                anchor_indices.append(anchor_idx)
                positive_indices.append(positive_idx)
                negative_indices.append(negative_idx)

        # Filter to keep only sets where the anchor belongs to the current GPU
        local_mask = [(start_idx <= a < end_idx) for a in anchor_indices]
        if not any(local_mask):
            # No valid sets for this GPU
            return torch.tensor(0.0, device=device, requires_grad=True)
        anchor_indices = [a for a, m in zip(anchor_indices, local_mask) if m]
        positive_indices = [p for p, m in zip(positive_indices, local_mask) if m]
        negative_indices = [n for n, m in zip(negative_indices, local_mask) if m]
        print(f'num sets: {len(anchor_indices)}')

        # Re-map memory bank positives after filtering
        old_to_new = {}
        new_set_idx = 0
        for i, m in enumerate(local_mask):
            if m:
                old_to_new[i] = new_set_idx
                new_set_idx += 1

        new_mb_positives = []
        for (old_set_idx, emb, lbl) in mb_positives:
            if old_set_idx in old_to_new:
                new_mb_positives.append((old_to_new[old_set_idx], emb, lbl))

        # Convert indices to tensors
        anchor_indices_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
        batch_positive_indices = [p for p in positive_indices if p >= 0]

        batch_positive_indices_t = (torch.tensor(batch_positive_indices, dtype=torch.long, device=device)
                                    if batch_positive_indices else None)

        negative_indices_t = torch.tensor(negative_indices, dtype=torch.long, device=device)

        # Gather corresponding logits
        anchor_logits = all_x[anchor_indices_t]
        negative_logits = all_x[negative_indices_t]
        # Handle positives:
        # For batch positives:
        if batch_positive_indices_t is not None:
            batch_positive_logits = all_x[batch_positive_indices_t]
        else:
            batch_positive_logits = None

        # For memory bank positives:
        # We have their logits directly, no grad. We'll insert them in place of -1 indices.
        final_positive_logits = []
        mb_pos_idx = 0
        batch_pos_idx = 0
        for p in positive_indices:
            if p >= 0:
                # from batch
                final_positive_logits.append(batch_positive_logits[batch_pos_idx])
                batch_pos_idx += 1
            else:
                # from memory bank
                # new_mb_positives is aligned with final sets
                # Retrieve corresponding mb positive
                mb_emb = new_mb_positives[mb_pos_idx][1]
                mb_pos_idx += 1
                final_positive_logits.append(mb_emb.to(device))  # move to device, no grad
        final_positive_logits = torch.stack(final_positive_logits, dim=0)

        # Sum the logits: anchor, positive, negative
        summed_logits = anchor_logits + final_positive_logits + negative_logits

        # Targets for these sets are the anchor's target labels
        target_indices = all_target[anchor_indices_t]

        # Apply standard cross-entropy on the summed logits
        loss = F.cross_entropy(summed_logits, target_indices)
        # Update the memory bank with this batch's logits and labels
        self.memory_bank.add_samples(x, target)

        return loss
