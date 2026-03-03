"""
Belief propagation on binary trees via Numba.

Provides exact posterior inference for the controlled setting generative model.
Use run_BP for discrete symbol inference (masked prediction) and
run_BP_diffusion for continuous diffusion conditioning (softmax field).
"""

import numpy as np
from numba import njit,jit
from multiprocessing import Pool, cpu_count
import torch


@njit
def generate_tree(l,q,leaves):
    '''
    Generate the upward and downward messages for a tree structure.
    
    Args:
        l (int): The depth of the tree.
        q (int): The number of states for each variable.
        leaves (np.ndarray): The leaf node messages.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The upward and downward messages.
    '''
    up_messages = np.zeros((l+1,2**l,q))
    down_messages = np.zeros((l+1,2**l,q))
    for i in range(l):
        for j in range(2**i):
            up_messages[i,j,:] = 1/q
            down_messages[i,j,:] = 1/q
    up_messages[l,:,:] = 1/q
    for j in range(2**l):
        down_messages[l,j,:] = leaves[j,:] # Add the prescribed leaves
    return up_messages, down_messages


@njit
def generate_tree_no_fields(l,q):
    '''
    Generate the upward and downward messages for a tree structure without prescribed leaves.

    Args:
        l (int): The depth of the tree.
        q (int): The number of states for each variable.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The upward and downward messages.
    '''
    up_messages = np.zeros((l+1,2**l,q))
    down_messages = np.zeros((l+1,2**l,q))
    for i in range(l):
        for j in range(2**i):
            up_messages[i,j,:] = 1/q
            down_messages[i,j,:] = 1/q
    up_messages[l,:,:] = 1/q
    for j in range(2**l):
        down_messages[l,j,:] = 1/q # Add the prescribed leaves
    return up_messages, down_messages


@njit
def set_fields(down_messages,leaves):
    """Overwrite the leaf-level downward messages with observed fields.

    Args:
        down_messages (np.ndarray): downward messages array, shape ``(l+1, 2^l, q)``.
        leaves (np.ndarray): observed leaf fields, shape ``(n, q)``.
    """
    n = leaves.shape[0]
    for j in range(n):
        down_messages[-1,j,:] = leaves[j,:] # Add the prescribed leaves


def get_P_xlevel_root(M,level):
    """Compute root-to-virtual-leaf transition matrices for factorized BP.

    For each of the ``2^level`` virtual leaves, multiplies the appropriate
    marginalized M_L / M_R matrices along the binary path from root to leaf.

    Args:
        M (np.ndarray): transition tensor, shape ``(q, q, q)``.
        level (int): number of factorized tree levels.

    Returns:
        np.ndarray: transition matrices, shape ``(q, q, 2^level)``.
    """
    M_L = np.sum(M,axis=2)
    M_R = np.sum(M,axis=1)
    q = M_L.shape[0]
    leaves_indices = np.arange(2**(level+1))[2**level:]
    leaves_indices_binary = [bin(leaves_indices[i])[2:][1:] for i in range(len(leaves_indices))]
    probs = np.empty((q,q,2**level))
    probs_prev = np.empty((q,q,2**level))
    for i in range(2**level):
        probs[:,:,i] = np.eye(q)
        probs_prev[:,:,i] = np.eye(q)
        for j in range(level):
            if leaves_indices_binary[i][j] == '0':
                probs[:,:,i] = probs_prev[:,:,i]@M_L
            else:
                probs[:,:,i] = probs_prev[:,:,i]@M_R
            probs_prev[:,:,:] = probs[:,:,:]
    return probs


@njit
def numba_prod(arr):
    """
    Computes the product along the first axis (axis=0) for a 2D NumPy array.
    """
    dim1, dim2 = arr.shape  # Get array dimensions
    result = np.ones(dim2, dtype=arr.dtype)  # Output array (1D of size dim2)

    for j in range(dim2):  # Loop over second dimension (columns)
        prod = 1.0  # Initialize product
        for i in range(dim1):  # Loop over first dimension (rows)
            prod *= arr[i, j]  # Multiply elements
        result[j] = prod  # Store the result

    return result


@njit
def update_messages(l,q,up_messages,down_messages,M,factorized_layers,probs):
    """Run one full BP sweep: upward pass (leaves to root), then downward pass (root to leaves).
    
    Supports factorized BP (BP_k) when factorized_layers > 0, which replaces
    the top k levels of the tree with independent single-factor-to-root channels.
    """
    # Pre-allocate temporary message buffers
    r_up = np.zeros(q)
    l_up = np.zeros(q)
    v_down = np.zeros(q)
    # Upward pass: aggregate children messages into parent messages
    for i in range(l-1,factorized_layers-1,-1):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            # Update the outgoing messages
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not contiguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        v_down[p1] += l_down[p2]*M[p1,p2,p3]*r_down[p3]
            down_messages[i,j,:] = v_down/np.sum(v_down)
    # Factorized BP: replace top k levels with product-of-singles approximation
    if factorized_layers > 0:
        down_messages_factorized = np.zeros((3,2**factorized_layers,q)) # Along axis = 1 have before and after the factor node and then the variable node
        down_messages_factorized[-1,:2**factorized_layers,:] = down_messages[factorized_layers,:2**factorized_layers,:]
        for j in range(2**factorized_layers): # Do each of the 2**k factor nodes updates
            for p1 in range(q):
                for p2 in range(q):
                    down_messages_factorized[1,j,p1] += probs[p1,p2,j]*down_messages_factorized[-1,j,p2]
            down_messages_factorized[1,j,:] = down_messages_factorized[1,j,:]/np.sum(down_messages_factorized[1,j,:])
        down_messages_factorized[0,0,:] = numba_prod(down_messages_factorized[1,:,:])/np.sum(numba_prod(down_messages_factorized[1,:,:]))
        down_messages[0,0,:] = down_messages_factorized[0,0,:]
        # Now go back down
        up_messages_factorized = np.zeros((3,2**factorized_layers,q))
        up_messages_factorized[0,0,:] = 1/q
        for j in range(2**factorized_layers):
            mask = np.arange(2**factorized_layers) != j
            up_messages_factorized[1,j,:] = numba_prod(down_messages_factorized[1,mask,:])*up_messages_factorized[0,0,:]
            up_messages_factorized[1,j,:] = up_messages_factorized[1,j,:]/np.sum(up_messages_factorized[1,j,:])
            for p1 in range(q):
                for p2 in range(q):
                    up_messages_factorized[-1,j,p1] += probs[p2,p1,j]*up_messages_factorized[1,j,p2]
            up_messages_factorized[-1,j,:] = up_messages_factorized[-1,j,:]/np.sum(up_messages_factorized[-1,j,:])
        up_messages[factorized_layers,:2**factorized_layers,:] = up_messages_factorized[-1,:,:]
    # Downward pass: propagate root belief back to leaves
    for i in range(factorized_layers,l):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            # Update the outgoing messages
            r_up[:] = 0
            l_up[:] = 0
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not contiguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        r_up[p1] += v_up[p2]*M[p2,p3,p1]*l_down[p3]
                        l_up[p1] += v_up[p2]*M[p2,p1,p3]*r_down[p3]
            up_messages[i+1,2*j,:] = l_up/np.sum(l_up)
            up_messages[i+1,2*j+1,:] = r_up/np.sum(r_up)
    return up_messages,down_messages


@njit
def compute_marginals(l,q,up_messages,down_messages):
    """Compute marginal distributions at every node by combining up/down messages.

    Args:
        l (int): tree depth.
        q (int): alphabet size.
        up_messages (np.ndarray): upward messages, shape ``(l+1, 2^l, q)``.
        down_messages (np.ndarray): downward messages, shape ``(l+1, 2^l, q)``.

    Returns:
        np.ndarray: normalised marginals, shape ``(l+1, 2^l, q)``.
    """
    marginals = np.empty((l+1,2**l,q))
    for i in range(l+1):
        for j in range(2**i):
            marginals[i,j,:] = up_messages[i,j,:]*down_messages[i,j,:]
            marginals[i,j,:] = marginals[i,j,:]/np.sum(marginals[i,j,:])
    return marginals


@njit
def get_freeEnergy(M,l,q,up_messages,down_messages,factorized_layers=0):
    """Compute the Bethe free energy from converged BP messages.

    Args:
        M (np.ndarray): transition tensor, shape ``(q, q, q)``.
        l (int): tree depth.
        q (int): alphabet size.
        up_messages (np.ndarray): upward messages.
        down_messages (np.ndarray): downward messages.
        factorized_layers (int): number of factorized levels (BP_k).

    Returns:
        float: Bethe free energy (in units of log_q).
    """
    if factorized_layers > 0:
        M_L = np.sum(M,axis=2)
        M_R = np.sum(M,axis=1)
    # First compute the free energy from the variables
    F_variables = 0
    for i in range(1,l): # Exclude both the root and the leaves
        for j in range(2**i):
            F_variables += np.log(np.sum(up_messages[i,j,:]*down_messages[i,j,:]))/np.log(q)
    # Now compute the free energy from the factors
    F_factors = 0
    for i in range(l):
        if i < factorized_layers:
            M_eff = np.empty((q,q,q))
            for j in range(q):
                M_eff[j,:,:] = np.outer(M_L[j,:],M_R[j,:])
        else:
            M_eff = M
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            z_factor = 0
            for p1 in range(q): # Not using @ because M matrix is not contiguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        z_factor += v_up[p1]*M_eff[p1,p2,p3]*l_down[p2]*r_down[p3]
            F_factors += np.log(z_factor)/np.log(q)
    return -(F_factors - F_variables)


def run_BP(M,l,q,xis,factorized_layers=0):
    """Run BP on a discrete sequence and return leaf marginals + free energy.

    Args:
        M (np.ndarray): transition tensor, shape ``(q, q, q)``.
        l (int): tree depth.
        q (int): alphabet size.
        xis (array-like): discrete leaf symbols in ``[0, q-1]`` or ``q+1``
            for masked positions.
        factorized_layers (int): number of factorized levels (BP_k).

    Returns:
        tuple[np.ndarray, float]: ``(marginals, free_energy)`` where
        *marginals* has shape ``(l+1, 2^l, q)``.
    """
    # Convert the leaves into messages, not super efficient but sequences are not so long and just need to do it once
    leaves_BP = np.empty((len(xis),q))
    for i in range(len(xis)):
        if xis[i] == q + 1: # Masked symbols
            leaves_BP[i,:] = 1/q
        else:
            leaves_BP[i,:] = 0
            leaves_BP[i,xis[i]] = 1
    up_messages,down_messages = generate_tree(l,q,leaves_BP)
    if factorized_layers > 0:
        probs = get_P_xlevel_root(M,factorized_layers)
        up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M,factorized_layers,probs)
    else:
        up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M,factorized_layers,np.ones((q,q,1)))
    freeEnergy = get_freeEnergy(M,l,q,up_messages,down_messages,factorized_layers)
    marginals = compute_marginals(l,q,up_messages,down_messages)
    return marginals,freeEnergy


def run_BP_diffusion(M,l,q,field,factorized_layers=0):
    """Run BP with continuous softmax fields and return leaf marginals.

    Used by the diffusion backward process to compute the exact denoiser
    score on the controlled setting.

    Args:
        M (np.ndarray): transition tensor, shape ``(q, q, q)``.
        l (int): tree depth.
        q (int): alphabet size.
        field (array-like): continuous field at the leaves, shape ``(2^l, q)``.
            Converted to probabilities via a numerically stable softmax.
        factorized_layers (int): number of factorized levels (BP_k).

    Returns:
        np.ndarray: leaf marginals, shape ``(2^l, q)``.
    """
    # Convert the leaves into messages, not super efficient but sequences are not so long and just need to do it once
    if isinstance(field, torch.Tensor):
        field = field.cpu().numpy()
    leaves_BP = np.empty(field.shape)
    for i in range(field.shape[0]):
        max_val = np.max(field[i, :])
        leaves_BP[i, :] = np.exp(field[i, :] - max_val) / np.sum(np.exp(field[i, :] - max_val))
    up_messages,down_messages = generate_tree(l,q,leaves_BP)
    if factorized_layers > 0:
        probs = get_P_xlevel_root(M,factorized_layers)
        up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M,factorized_layers,probs)
    else:
        up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M,factorized_layers,np.ones((q,q,1)))
    marginals = compute_marginals(l,q,up_messages,down_messages)
    return marginals[-1,:,:]