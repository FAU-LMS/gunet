import torch
import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import cg
from cupyx.scipy.sparse.linalg import spsolve
import scipy.sparse as sp

MAX_ITER = 150


def create_fixed_cupy_sparse_matrices_fixed_scale(H, W, upsampling):
    h = H // upsampling
    w = W // upsampling

    # create the mapping matrix from neighbor dense affinity to sparse Laplacian
    matrices = {}
    for location in ('top', 'bottom', 'left', 'right'):
        indices = np.zeros((4, H * W - W), dtype=int)
        l = 0
        for i, j in np.ndindex(H, W):
            if location == 'top' and i > 0:
                indices[:, l] = np.array([i, j, i - 1, j], dtype=int)
            elif location == 'bottom' and i < H - 1:
                indices[:, l] = np.array([i, j, i + 1, j], dtype=int)
            elif location == 'left' and j > 0:
                indices[:, l] = np.array([i, j, i, j - 1], dtype=int)
            elif location == 'right' and j < W - 1:
                indices[:, l] = np.array([i, j, i, j + 1], dtype=int)
            else:
                continue

            l += 1

        assert l == H * W - W
        indices_a, indices_b = np.unravel_index(np.ravel_multi_index(indices, (H, W, H, W)), (H * W, H * W))
        matrices[f'remap_{location}'] = cp.sparse.coo_matrix(sp.coo_matrix(
            (-1 * np.ones(len(indices_a), dtype=np.float32), (indices_a, indices_b)), shape=(H * W, H * W))).tocsr()

    matrices['remap_center'] = cp.sparse.eye(H * W, format='csr', dtype=np.float32)
    M = -matrices['remap_top'] - matrices['remap_bottom'] - matrices['remap_left'] - matrices['remap_right'] \
        + matrices['remap_center']

    # create array for downsampling
    D = cp.zeros((h * w, h, w), dtype=np.float32)
    for i, j in np.ndindex(h, w):
        D[i * w + j, i, j] = 1
    D = cp.kron(D, cp.ones((1, upsampling, upsampling), dtype=np.float32)) / (upsampling**2)  # h*w x H x W
    D = cp.sparse.coo_matrix(D.reshape((h * w, H * W))).tocsr()
    DtD = D.transpose().dot(D)
    
    return {**matrices, 'M': M, 'D': D, 'DtD': DtD}


def create_fixed_cupy_sparse_matrices(H, W, h, w):
    # create the mapping matrix from neighbor dense affinity to sparse Laplacian
    matrices = {}
    for location in ('top', 'bottom', 'left', 'right'):
        indices = np.zeros((4, H * W - W), dtype=int)
        l = 0
        for i, j in np.ndindex(H, W):
            if location == 'top' and i > 0:
                indices[:, l] = np.array([i, j, i - 1, j], dtype=int)
            elif location == 'bottom' and i < H - 1:
                indices[:, l] = np.array([i, j, i + 1, j], dtype=int)
            elif location == 'left' and j > 0:
                indices[:, l] = np.array([i, j, i, j - 1], dtype=int)
            elif location == 'right' and j < W - 1:
                indices[:, l] = np.array([i, j, i, j + 1], dtype=int)
            else:
                continue

            l += 1

        assert l == H * W - W
        indices_a, indices_b = np.unravel_index(np.ravel_multi_index(indices, (H, W, H, W)), (H * W, H * W))
        matrices[f'remap_{location}'] = cp.sparse.coo_matrix(sp.coo_matrix(
            (-1 * np.ones(len(indices_a), dtype=np.float32), (indices_a, indices_b)), shape=(H * W, H * W))).tocsr()

    matrices['remap_center'] = cp.sparse.eye(H * W, format='csr', dtype=np.float32)
    M = -matrices['remap_top'] - matrices['remap_bottom'] - matrices['remap_left'] - matrices['remap_right'] \
        + matrices['remap_center']

    # create array for downsampling
    D = cp.zeros((h, w, H, W))
    for y in range(h):
        for x in range(w):
            start_hr_y = y * H/h
            start_hr_x = x * W/w
            end_hr_y = (y + 1) * H/h
            end_hr_x = (x + 1) * W/w
            area_start_hr_y = int(cp.floor(start_hr_y))
            area_start_hr_x = int(cp.floor(start_hr_x))
            area_end_hr_y = int(cp.ceil(end_hr_y))
            area_end_hr_x = int(cp.ceil(end_hr_x))
            # Within these bounds some influence
            D[y, x, area_start_hr_y:area_end_hr_y, area_start_hr_x:area_end_hr_x] = 1
            # Partical cover
            upper_start_hr_y = int(cp.ceil(start_hr_y))
            upper_start_hr_x = int(cp.ceil(start_hr_x))
            lower_end_hr_y = int(cp.floor(end_hr_y))
            lower_end_hr_x = int(cp.floor(end_hr_x))

            lower_cover_y = upper_start_hr_y - start_hr_y
            lower_cover_x = upper_start_hr_x - start_hr_x
            upper_cover_y = end_hr_y - lower_end_hr_y
            upper_cover_x = end_hr_x - lower_end_hr_x
            if lower_cover_y != 0:
                D[y, x, area_start_hr_y, area_start_hr_x:area_end_hr_x] *= lower_cover_y
            if lower_cover_x != 0:
                D[y, x, area_start_hr_y:area_end_hr_y, area_start_hr_x] *= lower_cover_x
            if upper_cover_y != 0:
                D[y, x, area_end_hr_y - 1, area_start_hr_x:area_end_hr_x] *= upper_cover_y
            if upper_cover_x != 0:
                D[y, x, area_start_hr_y:area_end_hr_y, area_end_hr_x - 1] *= upper_cover_x
    D = D.reshape((h * w, H * W))
    D = D/cp.sum(D, axis=1)[:, None]
    D = cp.sparse.coo_matrix(D).tocsr()
    DtD = D.transpose().dot(D)

    return {**matrices, 'M': M, 'D': D, 'DtD': DtD}


class GraphQuadraticSolver(torch.autograd.Function):
    """
    Solves the problem min_x ||Dx - x_lr||^2 + lambda x^T L x, propagating gradients through the laplacian L.
    """

    @staticmethod
    def forward(ctx, neighbor_affinity, source, fixed_matrices):
        """
        neighbor_affinity (B x 5 x H x W): affinity among neighbor pixels
        source (B x 1 x h x w): source image
        """
        assert neighbor_affinity.is_cuda
        assert source.is_cuda
        assert source.shape[1] == 1
        assert not source.requires_grad #TODO

        B, _, h, w = source.shape

        D = fixed_matrices['D']
        DtD = fixed_matrices['DtD']
        neighbor_affinity_cp = cp.asarray(neighbor_affinity.detach())
        source_cp = cp.asarray(source.detach().reshape((B, -1, 1)))

        As = []
        bs = []
        for idx in range(0, B):
            L = build_laplacian(neighbor_affinity_cp[idx:idx + 1], fixed_matrices)
            As.append((DtD + L))
            bs.append(D.transpose().dot(source_cp[idx]))

        A = [[None] * B for _ in range(B)]
        for i in range(B):
            A[i][i] = As[i]
        A = cp.sparse.bmat(A).tocsr()
        b = cp.concatenate(bs, axis=0)

        x_cp = cg(A, b, maxiter=MAX_ITER)[0]
        #x_cp = spsolve(A, b)
        x_cp = x_cp.reshape((B, -1, 1))
        x = torch.as_tensor(x_cp, device='cuda')
        x = x.reshape((B, 1, neighbor_affinity.shape[2], neighbor_affinity.shape[3]))

        if ctx.needs_input_grad[0]:
            x.requires_grad = True

        ctx.x_cp = x_cp
        ctx.fixed_matrices = fixed_matrices
        ctx.A = A
        ctx.neighbor_affinity_shape = neighbor_affinity.shape

        return x

    @staticmethod
    def backward(ctx, grad_x):
        if grad_x is None:
            return None, None, None, None

        assert grad_x.shape[1] == 1

        x_cps = ctx.x_cp
        fixed_matrices = ctx.fixed_matrices 
        A = ctx.A
        B, _, H, W = grad_x.shape

        grad_x_cp = cp.asarray(grad_x.detach())
        grad_x_cp = grad_x_cp.reshape((-1, 1))
        grad_b_cp = cg(A.transpose().tocsr(), grad_x_cp, maxiter=MAX_ITER)[0]
        grad_b_cp = grad_b_cp.reshape((B, -1, 1))

        grad_neighbor_affinities = []
        for idx in range(0, B):
            grad_neighbor_affinity_cp = cp.zeros((5, A.shape[0] // B))

            for i, k in enumerate(('remap_top', 'remap_bottom', 'remap_left', 'remap_right')):
                grad_neighbor_affinity_cp[i] = (cp.sparse.diags(-grad_b_cp[idx].squeeze()).dot(
                    fixed_matrices[k].transpose().dot(cp.sparse.diags(x_cps[idx].squeeze())))).sum(axis=0).squeeze()

            grad_neighbor_affinity_cp[4] = (cp.sparse.diags(-grad_b_cp[idx].squeeze()).dot(
                fixed_matrices['remap_center'].dot(cp.sparse.diags(x_cps[idx].squeeze())))).sum(axis=0).squeeze()
            grad_neighbor_affinity = torch.as_tensor(grad_neighbor_affinity_cp, device='cuda')
            grad_neighbor_affinities.append(grad_neighbor_affinity.reshape((1, 5, H, W)))

        grad_neighbor_affinity = torch.cat(grad_neighbor_affinities, dim=0)
        return grad_neighbor_affinity, None, None, None


def build_laplacian(neighbor_affinity, remap):
    _, _, H, W = neighbor_affinity.shape
    neighbor_affinity = neighbor_affinity.reshape(5, H * W)

    return cp.sparse.diags(neighbor_affinity[0]).dot(remap['remap_top']) + \
        cp.sparse.diags(neighbor_affinity[1]).dot(remap['remap_bottom']) + \
        cp.sparse.diags(neighbor_affinity[2]).dot(remap['remap_left']) + \
        cp.sparse.diags(neighbor_affinity[3]).dot(remap['remap_right']) + \
        cp.sparse.diags(neighbor_affinity[4]).dot(remap['remap_center'])
