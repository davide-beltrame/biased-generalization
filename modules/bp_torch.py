import torch
import torch.nn as nn
import numpy as np


def get_P_xlevel_root_torch(M: torch.Tensor, level: int) -> torch.Tensor:
    """
    Compute transition probabilities from root to each 'virtual leaf' at factorized level.
    
    This is the PyTorch equivalent of belief_propagation.get_P_xlevel_root.
    For each of the 2^level factor nodes, computes the probability transition matrix
    P(x_leaf | x_root) by multiplying marginalized M matrices along the path.
    
    Args:
        M: Transition matrix (q, q, q) -> (Parent, Left, Right)
        level: Number of factorized layers (k in the paper)
        
    Returns:
        probs: (q, q, 2^level) tensor where probs[:, :, j] is the transition matrix
               from root state to virtual-leaf-j state
    """
    device = M.device
    dtype = M.dtype
    q = M.shape[0]
    
    # Marginalize M to get left and right transition matrices
    # M_L[parent, child] = sum_right M[parent, child, right] = P(left_child | parent)
    # M_R[parent, child] = sum_left M[parent, left, child] = P(right_child | parent)
    M_L = M.sum(dim=2)  # (q, q) - marginalize out right child
    M_R = M.sum(dim=1)  # (q, q) - marginalize out left child
    
    n_leaves = 2 ** level
    
    # Generate binary paths for each leaf: '0' = go left, '1' = go right
    # Leaf index i in range [2^level, 2^(level+1)) has binary representation
    # that encodes the path (excluding leading 1)
    leaves_indices = torch.arange(n_leaves, 2 * n_leaves, device=device)
    
    # Initialize: each leaf starts with identity (no transitions yet)
    probs = torch.zeros(q, q, n_leaves, device=device, dtype=dtype)
    
    for i in range(n_leaves):
        # Get binary path: convert to binary string, remove '0b1' prefix
        idx = leaves_indices[i].item()
        path_binary = bin(idx)[3:]  # Remove '0b1' to get path of length 'level'
        
        # Start with identity matrix
        current = torch.eye(q, device=device, dtype=dtype)
        
        # Multiply transition matrices along the path
        for step in path_binary:
            if step == '0':
                current = current @ M_L  # Go left
            else:
                current = current @ M_R  # Go right
        
        probs[:, :, i] = current
    
    return probs


class TreeBeliefPropagation(nn.Module):
    """Batched belief propagation on a binary tree in log-space (PyTorch).

    Drop-in GPU replacement for the Numba-based routines in
    ``belief_propagation.py``.  All message-passing is done in log-space
    for numerical stability.
    """

    def __init__(self, M, depth, q):
        """
        PyTorch implementation of Belief Propagation on a Tree.
        
        Args:
            M (Tensor): Transition matrix of shape (Q, Q, Q) -> (Parent, Left, Right).
            depth (int): Depth of the tree.
            q (int): Number of states.
        """
        super().__init__()
        self.depth = depth
        self.q = q
        
        # Store M both in log-space and linear-space
        # Log-space for stable message passing, linear for free energy
        self.register_buffer('log_M', torch.log(M + 1e-16))
        self.register_buffer('M', M.clone())

    def normalize(self, log_msg):
        """Normalize log-space messages so they sum to 1 in probability space.

        Args:
            log_msg (torch.Tensor): unnormalised log-probabilities, last dim is
                the state axis.

        Returns:
            torch.Tensor: normalised log-probabilities (same shape).
        """
        return log_msg - torch.logsumexp(log_msg, dim=-1, keepdim=True)

    def log_update_parent(self, log_msg_left, log_msg_right):
        """Compute upward message to a parent by marginalising over children.

        Args:
            log_msg_left (torch.Tensor): left-child log-messages, shape
                ``(B, N, Q)``.
            log_msg_right (torch.Tensor): right-child log-messages, shape
                ``(B, N, Q)``.

        Returns:
            torch.Tensor: parent log-messages, shape ``(B, N, Q)``.
        """
        # Shapes:
        # log_M:       (1, 1, Q, Q, Q)  [Dims: Parent, Left, Right]
        # log_msg_left:  (B, N, 1, Q, 1)  [Aligns with Left dim of M]
        # log_msg_right: (B, N, 1, 1, Q)  [Aligns with Right dim of M]
        
        # 1. Broadcast and Sum Energies (Product in Prob domain)
        joint_potential = (
            self.log_M.view(1, 1, self.q, self.q, self.q) + 
            log_msg_left.unsqueeze(2).unsqueeze(4) + 
            log_msg_right.unsqueeze(2).unsqueeze(3)
        )
        
        # 2. Marginalize (LogSumExp) over Left (dim 3) and Right (dim 4)
        return torch.logsumexp(joint_potential, dim=(3, 4))

    def log_update_left_child(self, log_msg_parent, log_msg_sibling_right):
        """Compute downward message to a left child.

        Marginalises over parent and right sibling.

        Args:
            log_msg_parent (torch.Tensor): parent log-messages, ``(B, N, Q)``.
            log_msg_sibling_right (torch.Tensor): right-sibling log-messages,
                ``(B, N, Q)``.

        Returns:
            torch.Tensor: left-child log-messages, ``(B, N, Q)``.
        """
        # Shapes:
        # log_M:               (1, 1, Q, Q, Q) [Parent, Left, Right]
        # log_msg_parent:      (B, N, Q, 1, 1) [Aligns with Parent dim]
        # log_msg_sibling_right: (B, N, 1, 1, Q) [Aligns with Right dim]
        
        joint_potential = (
            self.log_M.view(1, 1, self.q, self.q, self.q) + 
            log_msg_parent.unsqueeze(3).unsqueeze(4) + 
            log_msg_sibling_right.unsqueeze(2).unsqueeze(3)
        )
        
        # Marginalize over Parent (dim 2) and Right (dim 4) -> Result aligns with Left (dim 3)
        return torch.logsumexp(joint_potential, dim=(2, 4))

    def log_update_right_child(self, log_msg_parent, log_msg_sibling_left):
        """Compute downward message to a right child.

        Marginalises over parent and left sibling.

        Args:
            log_msg_parent (torch.Tensor): parent log-messages, ``(B, N, Q)``.
            log_msg_sibling_left (torch.Tensor): left-sibling log-messages,
                ``(B, N, Q)``.

        Returns:
            torch.Tensor: right-child log-messages, ``(B, N, Q)``.
        """
        # Shapes:
        # log_M:               (1, 1, Q, Q, Q) [Parent, Left, Right]
        # log_msg_parent:      (B, N, Q, 1, 1) [Aligns with Parent dim]
        # log_msg_sibling_left:  (B, N, 1, Q, 1) [Aligns with Left dim]
        
        joint_potential = (
            self.log_M.view(1, 1, self.q, self.q, self.q) + 
            log_msg_parent.unsqueeze(3).unsqueeze(4) + 
            log_msg_sibling_left.unsqueeze(2).unsqueeze(4)
        )
        
        # Marginalize over Parent (dim 2) and Left (dim 3) -> Result aligns with Right (dim 4)
        return torch.logsumexp(joint_potential, dim=(2, 3))

    def forward(self, leaves_log_probs, return_messages=False):
        """
        Full BP Pass.
        
        Args:
            leaves_log_probs: (B, 2^depth, Q) - Input fields/likelihoods in log-space.
            return_messages: If True, also return up/down messages for free energy computation.
            
        Returns:
            marginals (List[Tensor]): List of log-marginals for each depth level.
            If return_messages=True, also returns (up_messages, down_messages) in probability space.
        """
        B = leaves_log_probs.shape[0]
        device = leaves_log_probs.device
        
        # 1. Downward Pass (Leaves -> Root)
        # We store messages exiting each level 'd' going upwards to 'd-1'
        # NOTE: indexing here matches depth. msgs_from_leaves[d] contains messages at depth d.
        msgs_from_leaves = [None] * (self.depth + 1)
        msgs_from_leaves[self.depth] = self.normalize(leaves_log_probs)
        
        for d in range(self.depth - 1, -1, -1):
            # Get messages from children (next level)
            children = msgs_from_leaves[d + 1] # (B, 2^(d+1), Q)
            
            # Split into Left and Right children
            # Reshape to (B, 2^d, 2, Q)
            children_grouped = children.view(B, -1, 2, self.q)
            left_child = children_grouped[:, :, 0, :]
            right_child = children_grouped[:, :, 1, :]
            
            # Compute message to parent
            msg_to_parent = self.log_update_parent(left_child, right_child)
            msgs_from_leaves[d] = self.normalize(msg_to_parent)

        # 2. Upward Pass (Root -> Leaves)
        msgs_from_root = [None] * (self.depth + 1)
        
        # Root initialization (Uniform prior in log space is 0 for unnormalized logits, 
        # or -log(Q) for normalized. Since we normalize at every step, 0 is fine).
        msgs_from_root[0] = torch.zeros(B, 1, self.q, device=device)
        
        for d in range(self.depth):
            parent = msgs_from_root[d] # (B, 2^d, Q)
            
            # Get siblings from the previous pass to help update
            children_from_leaves = msgs_from_leaves[d + 1].view(B, -1, 2, self.q)
            left_from_leaves = children_from_leaves[:, :, 0, :]
            right_from_leaves = children_from_leaves[:, :, 1, :]
            
            # Calculate messages to children
            to_left = self.log_update_left_child(parent, right_from_leaves)
            to_right = self.log_update_right_child(parent, left_from_leaves)
            
            # Combine back to (B, 2^(d+1), Q) and Interleave
            next_level = torch.stack([to_left, to_right], dim=2) # (B, 2^d, 2, Q)
            msgs_from_root[d+1] = self.normalize(next_level.view(B, -1, self.q))

        # 3. Compute Marginals
        marginals = []
        for d in range(self.depth + 1):
            log_belief = msgs_from_leaves[d] + msgs_from_root[d]
            marginals.append(self.normalize(log_belief))
        
        if return_messages:
            # Convert to probability space and return with numpy-compatible naming:
            # In numpy BP code:
            #   - "down_messages" = evidence aggregated from leaves (toward root)
            #   - "up_messages" = prior/context from root (toward leaves)
            # Our naming:
            #   - msgs_from_leaves = evidence from leaves = numpy's down_messages
            #   - msgs_from_root = prior from root = numpy's up_messages
            # Return tuple: (numpy_up_messages, numpy_down_messages) for free energy computation
            # NOTE: we normalize after exp() to match numpy's normalized messages
            numpy_up_messages = []
            numpy_down_messages = []
            for d in range(self.depth + 1):
                # msgs_from_root -> numpy's up_messages (normalize in prob space)
                probs_up = torch.exp(msgs_from_root[d])
                probs_up = probs_up / probs_up.sum(dim=-1, keepdim=True)
                numpy_up_messages.append(probs_up)
                
                # msgs_from_leaves -> numpy's down_messages (normalize in prob space)
                probs_down = torch.exp(msgs_from_leaves[d])
                probs_down = probs_down / probs_down.sum(dim=-1, keepdim=True)
                numpy_down_messages.append(probs_down)
            
            return marginals, (numpy_up_messages, numpy_down_messages)
            
        return marginals

    def forward_factorized(
        self,
        leaves_log_probs: torch.Tensor,
        probs: torch.Tensor,
        factorized_layers: int,
        return_messages: bool = False,
    ):
        """
        Factorized BP Pass - treats top k layers as a single 'super-factor'.
        
        Mirrors the logic of belief_propagation.update_messages (now refactored/removed) with factorized_layers > 0.
        
        Args:
            leaves_log_probs: (B, 2^depth, Q) - Input fields/likelihoods in log-space.
            probs: (Q, Q, 2^k) - Transition matrices from get_P_xlevel_root_torch.
            factorized_layers: k, the number of top layers to factorize.
            return_messages: If True, also return messages for free energy computation.
            
        Returns:
            marginals (List[Tensor]): List of log-marginals for each depth level.
        """
        B = leaves_log_probs.shape[0]
        device = leaves_log_probs.device
        k = factorized_layers
        n_k = 2 ** k  # Number of virtual leaves at factorized level
        
        # Initialize message arrays
        msgs_from_leaves = [None] * (self.depth + 1)
        msgs_from_root = [None] * (self.depth + 1)
        
        # PHASE 1: Standard down-pass from leaves up to level k
        msgs_from_leaves[self.depth] = self.normalize(leaves_log_probs)
        
        for d in range(self.depth - 1, k - 1, -1):  # Stop at level k (not going above)
            children = msgs_from_leaves[d + 1]
            children_grouped = children.view(B, -1, 2, self.q)
            left_child = children_grouped[:, :, 0, :]
            right_child = children_grouped[:, :, 1, :]
            msg_to_parent = self.log_update_parent(left_child, right_child)
            msgs_from_leaves[d] = self.normalize(msg_to_parent)
        
        # PHASE 2: Factorized root update
        # At level k, we have 2^k nodes with down-messages.
        # Each passes through the "super-factor" using probs[:,:,j].
        
        # down_messages_factorized: messages at the k-level nodes (B, 2^k, q)
        down_at_k = msgs_from_leaves[k]  # (B, 2^k, q) in log-space
        
        # Convert to probability space for factorized operations
        down_at_k_prob = torch.exp(down_at_k)
        down_at_k_prob = down_at_k_prob / down_at_k_prob.sum(dim=-1, keepdim=True)
        
        # Apply transition through k layers: multiply by probs[:,:,j].T
        # probs is (q, q, 2^k) where probs[root_state, leaf_state, leaf_idx]
        # We want: for each j, new_msg[b,j,p1] = sum_p2 probs[p1,p2,j] * down_at_k[b,j,p2]
        # This is: new_msg = einsum('pqj,bjq->bjp', probs, down_at_k_prob)
        down_after_factor = torch.einsum('pqj,bjq->bjp', probs, down_at_k_prob)
        down_after_factor = down_after_factor / down_after_factor.sum(dim=-1, keepdim=True)
        
        # Combine all messages at root via product
        # root_msg = prod_j down_after_factor[b,j,:]
        root_msg = down_after_factor.prod(dim=1)  # (B, q)
        root_msg = root_msg / root_msg.sum(dim=-1, keepdim=True)
        
        # Store root message (level 0)
        msgs_from_leaves[0] = torch.log(root_msg.unsqueeze(1) + 1e-16)
        
        # PHASE 3: Factorized up-pass
        # Start with uniform prior at root
        msgs_from_root[0] = torch.zeros(B, 1, self.q, device=device)  # log-uniform
        up_root_prob = torch.ones(B, 1, self.q, device=device) / self.q
        
        # Compute cavity messages (all-but-one product) at the super-factor
        # For each j: cavity_j = (root_prior * prod_{i != j} down_after_factor[i]) 
        # = (root_prior * root_msg / down_after_factor[j])
        # NOTE: root_msg = prod_j down_after_factor[j], so cavity = root_prior * root_msg / down_after_factor[j]
        
        # Expand: root_msg (B, q) -> (B, 1, q)
        root_msg_expanded = root_msg.unsqueeze(1)  # (B, 1, q)
        # Cavity: (B, 2^k, q)
        cavity = up_root_prob * root_msg_expanded / (down_after_factor + 1e-16)  # broadcast over j
        cavity = cavity / cavity.sum(dim=-1, keepdim=True)
        
        # Now propagate cavity messages back through probs.T to level k
        # up_at_k[b,j,p1] = sum_p2 probs[p2,p1,j] * cavity[b,j,p2]
        # = einsum('qpj,bjq->bjp', probs, cavity)  -- note index swap for transpose
        up_at_k_prob = torch.einsum('qpj,bjq->bjp', probs, cavity)
        up_at_k_prob = up_at_k_prob / up_at_k_prob.sum(dim=-1, keepdim=True)
        
        # Store at level k in log-space
        msgs_from_root[k] = torch.log(up_at_k_prob + 1e-16)
        
        # PHASE 4: Standard up-pass from level k to leaves
        for d in range(k, self.depth):
            parent = msgs_from_root[d]
            children_from_leaves = msgs_from_leaves[d + 1].view(B, -1, 2, self.q)
            left_from_leaves = children_from_leaves[:, :, 0, :]
            right_from_leaves = children_from_leaves[:, :, 1, :]
            
            to_left = self.log_update_left_child(parent, right_from_leaves)
            to_right = self.log_update_right_child(parent, left_from_leaves)
            
            next_level = torch.stack([to_left, to_right], dim=2)
            msgs_from_root[d + 1] = self.normalize(next_level.view(B, -1, self.q))
        
        # Only compute marginals for levels where we have both messages
        marginals = [None] * (self.depth + 1)
        
        # Level 0 (root)
        marginals[0] = self.normalize(msgs_from_leaves[0] + msgs_from_root[0])
        
        # Levels k to depth (have both messages from standard passes)
        for d in range(k, self.depth + 1):
            log_belief = msgs_from_leaves[d] + msgs_from_root[d]
            marginals[d] = self.normalize(log_belief)
        
        # Levels 1 to k-1: messages not directly computed
        for d in range(1, k):
            # These intermediate levels don't have proper messages in factorized BP
            # We can approximate as uniform or leave as None
            n_nodes = 2 ** d
            marginals[d] = torch.zeros(B, n_nodes, self.q, device=device) - np.log(self.q)
        
        if return_messages:
            numpy_up_messages = []
            numpy_down_messages = []
            for d in range(self.depth + 1):
                if msgs_from_root[d] is not None:
                    probs_up = torch.exp(msgs_from_root[d])
                    probs_up = probs_up / probs_up.sum(dim=-1, keepdim=True)
                else:
                    probs_up = torch.ones(B, 2**d, self.q, device=device) / self.q
                numpy_up_messages.append(probs_up)
                
                if msgs_from_leaves[d] is not None:
                    probs_down = torch.exp(msgs_from_leaves[d])
                    probs_down = probs_down / probs_down.sum(dim=-1, keepdim=True)
                else:
                    probs_down = torch.ones(B, 2**d, self.q, device=device) / self.q
                numpy_down_messages.append(probs_down)
            
            return marginals, (numpy_up_messages, numpy_down_messages)
        
        return marginals


def compute_free_energy_torch(up_messages, down_messages, M, depth, q):
    """
    Compute Bethe free energy from BP messages (torch version).
    
    This matches the numpy version in belief_propagation.get_freeEnergy.
    
    Free energy formula:
        F = -(F_factors - F_variables)
        
    where:
        F_variables = sum over internal nodes (depth 1 to depth-1) of 
                      log(sum(up * down)) / log(q)
        F_factors = sum over all factors of log(z_factor) / log(q)
            where z_factor = sum_{p1,p2,p3} up[p1] * M[p1,p2,p3] * l_down[p2] * r_down[p3]
    
    Message naming convention:
        - up_messages: messages from ROOT toward LEAVES (prior/context)
        - down_messages: messages from LEAVES toward ROOT (evidence)
    
    Args:
        up_messages: List of (B, 2^d, q) tensors - messages from root toward leaves
        down_messages: List of (B, 2^d, q) tensors - messages from leaves toward root (evidence)
        M: (q, q, q) transition matrix
        depth: Tree depth (l)
        q: Number of states
        
    Returns:
        free_energy: (B,) tensor of free energies

    """
    device = up_messages[0].device
    B = up_messages[0].shape[0]
    log_q = np.log(q)
    
    # Ensure M is on the correct device
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float().to(device)
    elif M.device != device:
        M = M.to(device)
    
    # F_variables: sum over internal nodes (depth 1 to depth-1) 
    # Internal nodes are those that are neither the root (depth 0) nor the leaves (depth l)
    # For each internal node: log(sum(up * down)) / log(q)
    F_variables = torch.zeros(B, device=device)
    
    for d in range(1, depth):  # Exclude root (d=0) and leaves (d=depth)
        # up_messages[d] and down_messages[d] have shape (B, 2^d, q)
        up = up_messages[d]  # (B, 2^d, q) - from root
        down = down_messages[d]  # (B, 2^d, q) - from leaves
        
        # Sum over q dimension: sum(up * down) for each node
        z_node = torch.sum(up * down, dim=-1)  # (B, 2^d)
        
        # Sum log(z_node) over all nodes at this depth
        # Clamp to avoid log(0)
        log_z = torch.log(z_node.clamp(min=1e-16))  # (B, 2^d)
        F_variables += log_z.sum(dim=-1) / log_q  # (B,)
    
    # F_factors: sum over all factors
    # Each factor connects a parent at depth d to its two children at depth d+1
    # For factor at (d, j):
    #   - v_up = up_messages[d, j] = message TO the factor from root (at parent node)
    #   - l_down = down_messages[d+1, 2j] = message TO the factor from left child
    #   - r_down = down_messages[d+1, 2j+1] = message TO the factor from right child
    # z_factor = sum_{p1,p2,p3} v_up[p1] * M[p1,p2,p3] * l_down[p2] * r_down[p3]
    F_factors = torch.zeros(B, device=device)
    
    for d in range(depth):  # depth 0 to depth-1
        # Get up message for parent nodes at depth d (from root direction)
        v_up = up_messages[d]  # (B, 2^d, q)
        
        # Get down messages for children at depth d+1 (from leaves/evidence)
        children_down = down_messages[d + 1]  # (B, 2^(d+1), q)
        
        # Reshape children to pair left/right: (B, 2^d, 2, q)
        children_grouped = children_down.view(B, -1, 2, q)
        l_down = children_grouped[:, :, 0, :]  # (B, 2^d, q) - left child
        r_down = children_grouped[:, :, 1, :]  # (B, 2^d, q) - right child
        
        # Compute z_factor = sum_{p1,p2,p3} up[p1] * M[p1,p2,p3] * l_down[p2] * r_down[p3]
        # Using einsum for efficiency:
        # M: (q, q, q) = [parent, left, right]
        # v_up: (B, N, q) where N = 2^d
        # l_down: (B, N, q)
        # r_down: (B, N, q)
        # Result: (B, N) where each entry is sum over p1,p2,p3
        z_factor = torch.einsum('bnp,plr,bnl,bnr->bn', v_up, M, l_down, r_down)  # (B, 2^d)
        
        # Sum log(z_factor) over all factors at this depth
        log_z = torch.log(z_factor.clamp(min=1e-16))  # (B, 2^d)
        F_factors += log_z.sum(dim=-1) / log_q  # (B,)
    
    # Final free energy: F = -(F_factors - F_variables)
    free_energy = -(F_factors - F_variables)
    
    return free_energy


def run_BP_diffusion_torch(M, depth, q, field, factorized_layers=0):
    """
    Drop-in replacement for belief_propagation.run_BP_diffusion.
    
    Args:
        M: Transition matrix (Q, Q, Q) - numpy or tensor
        depth: Tree depth (l)
        q: Number of states (vocab_size)
        field: Input field of shape (2^depth, Q) or (B, 2^depth, Q)
        factorized_layers: Number of top layers to treat as factorized (0 = exact BP)
        
    Returns:
        Leaf marginals: (2^depth, Q) or (B, 2^depth, Q) in probability space
    """
    # Handle numpy inputs
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float()
    if isinstance(field, np.ndarray):
        field = torch.from_numpy(field).float()
    
    # Ensure 3D: (B, N, Q)
    squeeze_output = False
    if field.dim() == 2:
        field = field.unsqueeze(0)
        squeeze_output = True
    
    # Field is already in log-like space (unnormalized log-probs)
    leaves_log_probs = field
    
    # Move to same device as input
    device = field.device
    M = M.to(device)
    
    # Create BP module and run
    bp = TreeBeliefPropagation(M, depth, q)
    bp.eval()
    bp = bp.to(device)
    
    with torch.no_grad():
        if factorized_layers == 0:
            # Standard exact BP
            marginals_list = bp(leaves_log_probs)
        else:
            # Factorized BP
            probs = get_P_xlevel_root_torch(M, factorized_layers)
            marginals_list = bp.forward_factorized(
                leaves_log_probs, probs, factorized_layers
            )
    
    # Return leaf marginals in probability space (matching original interface)
    leaf_marginals = torch.exp(marginals_list[-1])
    
    if squeeze_output:
        leaf_marginals = leaf_marginals.squeeze(0)
    
    return leaf_marginals


def run_BP_torch(M, depth, q, xis, factorized_layers=0):
    """
    Full BP with free energy computation - torch version of belief_propagation.run_BP.
    
    Args:
        M: Transition matrix (Q, Q, Q) - numpy or tensor
        depth: Tree depth (l)
        q: Number of states (vocab_size)
        xis: Integer sequence (2^depth,) or batch (B, 2^depth) with values in [0, q-1] or q+1 for masked
        factorized_layers: Number of top layers to treat as factorized (0 = exact BP)
        
    Returns:
        marginals: Leaf marginals (2^depth, Q) or (B, 2^depth, Q) in probability space
        free_energy: Scalar or (B,) free energy values
    """
    # Handle numpy inputs
    if isinstance(M, np.ndarray):
        M_tensor = torch.from_numpy(M).float()
    else:
        M_tensor = M.clone()
    
    if isinstance(xis, np.ndarray):
        xis = torch.from_numpy(xis).long()
    
    # Ensure 2D: (B, N)
    squeeze_output = False
    if xis.dim() == 1:
        xis = xis.unsqueeze(0)
        squeeze_output = True
    
    B, N = xis.shape
    device = xis.device
    
    # Move M to same device
    M_tensor = M_tensor.to(device)
    
    # Convert integer sequences to one-hot leaf probabilities
    # xis[i] == q+1 means masked -> uniform
    leaves_probs = torch.zeros(B, N, q, device=device, dtype=M_tensor.dtype)
    mask = (xis == q + 1)
    non_mask = ~mask
    
    # Masked positions get uniform distribution
    leaves_probs[mask] = 1.0 / q
    
    # Non-masked positions get one-hot
    # Clamp indices to valid range for non-masked
    valid_indices = xis.clamp(0, q - 1)
    leaves_probs[non_mask] = 0.0
    leaves_probs.scatter_(2, valid_indices.unsqueeze(-1), 1.0)
    # Re-apply uniform for masked (scatter may have overwritten)
    leaves_probs[mask] = 1.0 / q
    
    # Convert to log space
    leaves_log_probs = torch.log(leaves_probs + 1e-16)
    
    # Create BP module and run
    bp = TreeBeliefPropagation(M_tensor, depth, q)
    bp.eval()
    bp = bp.to(device)
    
    with torch.no_grad():
        if factorized_layers == 0:
            # Standard exact BP
            marginals_list, (up_messages, down_messages) = bp(leaves_log_probs, return_messages=True)
        else:
            # Factorized BP
            probs = get_P_xlevel_root_torch(M_tensor, factorized_layers)
            marginals_list, (up_messages, down_messages) = bp.forward_factorized(
                leaves_log_probs, probs, factorized_layers, return_messages=True
            )
    
    # Compute free energy from probability-space messages
    free_energy = compute_free_energy_torch(up_messages, down_messages, M_tensor, depth, q)
    
    # Return leaf marginals in probability space
    leaf_marginals = torch.exp(marginals_list[-1])
    
    if squeeze_output:
        leaf_marginals = leaf_marginals.squeeze(0)
        free_energy = free_energy.squeeze(0)
    
    return leaf_marginals, free_energy


def compute_bp_free_energies_torch(sequences, M, l, q, batch_size=256):
    """
    Compute free energies for a batch of sequences using belief propagation (torch version).
    
    This is a batched GPU implementation, much faster than the numpy version for large datasets.
    
    Args:
        sequences: (n_sequences, seq_len) integer array of discrete sequences
        M: (q, q, q) transition tensor
        l: Tree depth
        q: Vocabulary size
        batch_size: Batch size for processing (default 256)
        
    Returns:
        free_energies: List of free energies (one per sequence)
    """
    from tqdm import tqdm
    
    # Convert inputs to tensors
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences).long()
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences = sequences.to(device)
    M = M.to(device)
    
    n_sequences = sequences.shape[0]
    free_energies = []
    
    # Process in batches
    for i in tqdm(range(0, n_sequences, batch_size), desc="Computing BP free energies (torch)"):
        batch = sequences[i:i + batch_size]
        _, batch_fe = run_BP_torch(M, l, q, batch, factorized_layers=0)
        free_energies.extend(batch_fe.cpu().tolist())
    
    return free_energies