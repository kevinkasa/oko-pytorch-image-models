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

    def __init__(self, memory_bank: MemoryBank, hardness_method: str = 'prob'):
        super().__init__()
        self.memory_bank = memory_bank
        self.hardness_method = hardness_method
        print(f'Hardness measure: {self.hardness_method}')

    def __compute_hardness_scores(
            self,
            anchor_label: int,
            anchor_logits: torch.Tensor,
            anchor_embedding: torch.Tensor,
            negative_logits: torch.Tensor,
            negative_embeddings: torch.Tensor,
            method: str = 'prob'
    ):
        if method == 'prob':
            probs = F.softmax(negative_logits, dim=-1)
            probs_except_anchor = probs.clone()
            probs_except_anchor[:, anchor_label] = -float('inf')
            hardness_scores = torch.max(probs_except_anchor, dim=-1)[0]

        elif method == 'cdist':
            anchor_embedding_expanded = anchor_embedding.unsqueeze(0)
            dists = torch.cdist(anchor_embedding_expanded, negative_embeddings).squeeze(0)
            hardness_scores = dists

        elif method == 'cosine':
            anchor_expanded = anchor_embedding.unsqueeze(0).expand(negative_embeddings.size(0), -1)
            sim = F.cosine_similarity(anchor_expanded, negative_embeddings, dim=-1)
            hardness_scores = 1 - sim

        elif method == 'kl':
            prob_anchor = F.softmax(anchor_logits.unsqueeze(0), dim=-1)
            prob_negative = F.softmax(negative_logits, dim=-1)
            kl_values = F.kl_div(prob_anchor.log().expand(prob_negative.size(0), -1),
                                 prob_negative, reduction='none').sum(dim=-1)
            hardness_scores = kl_values
        else:
            raise ValueError(f"Unknown hardness method: {method}")

        return hardness_scores

    def forward(self, x: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
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

            # Gather embeddings
            all_embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
            dist.all_gather(all_embeddings_list, embeddings)
            all_embeddings = torch.cat(all_embeddings_list, dim=0)
        else:
            # Not distributed
            all_x = x
            all_target = target
            all_embeddings = embeddings

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

            # Hard negative mining from the current batch
            anchor_logits = all_x[anchor_idx]
            anchor_embedding = all_embeddings[anchor_idx]

            negative_logits_candidates = all_x[negative_pool]  # [num_negatives, num_classes]
            negative_embeddings_candidates = all_embeddings[negative_pool]
            hardness_scores = self.__compute_hardness_scores(
                anchor_label=anchor_label,
                anchor_logits=anchor_logits,
                anchor_embedding=anchor_embedding,
                negative_logits=negative_logits_candidates,
                negative_embeddings=negative_embeddings_candidates,
                method=self.hardness_method
            )

            hardest_idx = torch.argmax(hardness_scores).item()
            hardest_negative = negative_pool[hardest_idx]
            negative_indices.append(hardest_negative)

            # Compute hardness: For each candidate in negative_pool, pick the one
            # that gives the highest probability to anchor_label.
            # negative_logits_candidates = all_x[negative_pool]  # [num_candidates, num_classes]
            # Compute softmax probabilities
            # probs = F.softmax(negative_logits_candidates, dim=-1)
            # probs_except_anchor = probs.clone()
            # probs_except_anchor[:, anchor_label] = -float('inf')
            # hardness_scores = torch.max(probs_except_anchor, dim=-1)[0]
            # hardest_idx = torch.argmax(hardness_scores).item()
            # hardest_negative = negative_pool[hardest_idx]

            # # Hardness = probability of anchor_label
            # hardness_scores = probs[:, anchor_label]
            # hardest_idx = torch.argmax(hardness_scores).item()
            # hardest_negative = negative_pool[hardest_idx]

            # # Add the chosen negative
            # negative_indices.append(hardest_negative)

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


class OKOAllTripletsLimited(nn.Module):
    """
    Forms ALL valid triplets from the current global batch, but then
    limits the total number of sets to a specified `max_sets` (if more are formed).
    """

    def __init__(self, max_sets: int = 512):
        """
        Args:
            max_sets (int): Maximum number of triplets to sample per forward pass.
        """
        super().__init__()
        self.max_sets = max_sets

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Logits of shape [local_batch_size, num_classes].
            target (torch.Tensor): Class labels of shape [local_batch_size].
        """
        device = x.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # --- 1) Gather logits and targets across GPUs ---
        if world_size > 1:
            # Use the custom all_gather with gradient function for logits
            gathered_list = list(all_gather_with_grad(x))
            # Replace the current GPU's portion with the original x that has grad_fn
            gathered_list[rank] = x  # restore local grad_fn
            all_x = torch.cat(gathered_list, dim=0)

            # Gather targets using standard all_gather (no grad needed for targets)
            all_target_list = [torch.zeros_like(target) for _ in range(world_size)]
            dist.all_gather(all_target_list, target)
            all_target = torch.cat(all_target_list, dim=0)
        else:
            all_x = x
            all_target = target

        global_batch_size = all_x.size(0)
        all_targets_np = all_target.cpu().tolist()

        # Build a mapping: label -> list of sample indices
        label_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(all_targets_np):
            label_to_indices.setdefault(lbl, []).append(idx)

        # --- 2) Build all triplets (anchor, positive, negative) in a vectorized manner ---
        # We'll accumulate sets in a list-of-tensors and then cat at the end.
        triplets_list = []
        all_indices = torch.arange(global_batch_size, device=device)
        for c, indices_c in label_to_indices.items():
            if len(indices_c) < 2:
                # Can't form pairs from this class, skip
                continue

            # Convert to a PyTorch tensor for combination ops
            indices_c_t = torch.tensor(indices_c, device=device, dtype=torch.long)

            # (a) All distinct pairs of this class (anchor, positive)
            pairs = torch.combinations(indices_c_t, r=2,with_replacement =False)  # shape: [n_pairs, 2]

            # (b) Negatives: gather all indices not in this class
            neg_mask = torch.ones(global_batch_size, dtype=torch.bool, device=device)
            neg_mask[indices_c_t] = False
            neg_indices = all_indices[neg_mask]  # shape [n_neg]
            if neg_indices.numel() == 0:
                continue  # no negative

            n_pairs = pairs.size(0)
            n_neg = neg_indices.size(0)

            pairs_expand = pairs.repeat_interleave(n_neg, dim=0)  # shape: [n_pairs * n_neg, 2]
            neg_expand = neg_indices.repeat(n_pairs)  # shape: [n_pairs * n_neg]
            triplets = torch.stack((pairs_expand[:, 0], pairs_expand[:, 1], neg_expand), dim=1)
            triplets_list.append(triplets)

        if not triplets_list:
            # No valid triplets
            return torch.tensor(0.0, device=device, requires_grad=True)

        all_triplets = torch.cat(triplets_list, dim=0)  # shape [N, 3]

        # --- 3) Filter triplets so anchor belongs to local rank ---
        local_batch_size = x.size(0)
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size

        anchor_indices = all_triplets[:, 0]
        local_mask = (anchor_indices >= start_idx) & (anchor_indices < end_idx)
        if not local_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        valid_triplets = all_triplets[local_mask]  # shape [M, 3]

        # --- 4) Limit the number of triplets to `max_sets` ---
        num_triplets = valid_triplets.size(0)
        if num_triplets > self.max_sets:
            # We can sample randomly or take the first self.max_sets
            # For best coverage, random sample is typical
            chosen_indices = torch.randperm(num_triplets, device=device)[:self.max_sets]
            valid_triplets = valid_triplets[chosen_indices]

        # --- 5) Sum the 3 logits and compute cross-entropy ---
        anchor_logits = all_x[valid_triplets[:,0]]
        positive_logits = all_x[valid_triplets[:,1]]
        negative_logits = all_x[valid_triplets[:,2]]

        summed_logits = anchor_logits + positive_logits + negative_logits
        anchor_labels = all_target[valid_triplets[:,0]]

        loss = F.cross_entropy(summed_logits, anchor_labels)
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
        # summed_logits = (anchor_logits + final_positive_logits + negative_logits) / 3
        # Targets for these sets are the anchor's target labels
        target_indices = all_target[anchor_indices_t]

        # Apply standard cross-entropy on the summed logits
        loss = F.cross_entropy(summed_logits, target_indices)
        # Update the memory bank with this batch's logits and labels
        self.memory_bank.add_samples(x, target)

        return loss
