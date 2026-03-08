import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import label
from scipy import sparse
from scipy.sparse.linalg import spsolve

device = 'cuda:2'

# ══════════════════════════════════════════════════════════════════════════════
#  Problem definitions
# ══════════════════════════════════════════════════════════════════════════════

class Problems():
    def dlX_disp(self):
        n = self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0]
        domain_xcoord = np.random.uniform(
            -self.nelx / (2 * self.nelm), self.nelx / (2 * self.nelm), (n, 1))
        domain_ycoord = np.random.uniform(
            -self.nely / (2 * self.nelm), self.nely / (2 * self.nelm), (n, 1))
        domain_zcoord = np.random.uniform(
            -self.nelz / (2 * self.nelm), self.nelz / (2 * self.nelm), (n, 1))
        domain_coord = np.concatenate((domain_ycoord, domain_xcoord, domain_zcoord), axis=1)
        coord = np.concatenate(
            (self.dlX_fixed.cpu().detach().numpy(),
             self.dlX_force.cpu().detach().numpy()), axis=0)
        coord = np.concatenate((coord, domain_coord), axis=0)
        return torch.tensor(coord, dtype=torch.float32, requires_grad=True).to(device)


class Cantilever_Beam_3D(Problems):
    def __init__(self, nelx, nely, nelz, xid, yid, zid, vf):
        self.xid  = xid;  self.yid  = yid;  self.zid  = zid
        self.nelx = nelx; self.nely = nely;  self.nelz = nelz
        self.nele = nelx * nely * nelz
        self.nelm = max(nelx, nely, nelz)
        self.volfrac = vf
        self.E0  = 1000; self.nu = 0.3
        self.batch_size  = 30000
        self.alpha_init  = 1; self.alpha_max = 100; self.alpha_delta = 0.5
        self.penal = 3.0

        c_y, c_x, c_z = np.meshgrid(
            np.linspace(-nely/(2*self.nelm), nely/(2*self.nelm), nely),
            np.linspace(-nelx/(2*self.nelm), nelx/(2*self.nelm), nelx),
            np.linspace(-nelz/(2*self.nelm), nelz/(2*self.nelm), nelz),
            indexing='ij')
        self.dlX = np.stack(
            (c_y.reshape([-1]), c_x.reshape([-1]), c_z.reshape([-1])), axis=1
        ).reshape([-1, 3])

        c_y, c_x, c_z = np.meshgrid(
            np.linspace(-nely/(2*self.nelm), nely/(2*self.nelm), 2*nely),
            np.linspace(-nelx/(2*self.nelm), nelx/(2*self.nelm), 2*nelx),
            np.linspace(-nelz/(2*self.nelm), nelz/(2*self.nelm), 2*nelz),
            indexing='ij')
        self.dlXSS = np.stack(
            (c_y.reshape([-1]), c_x.reshape([-1]), c_z.reshape([-1])), axis=1
        ).reshape([-1, 3])

        self.V = (
            (np.max(self.dlX[:, 0]) - np.min(self.dlX[:, 0])) *
            (np.max(self.dlX[:, 1]) - np.min(self.dlX[:, 1])) *
            (np.max(self.dlX[:, 2]) - np.min(self.dlX[:, 2])))

        fixed_voxel = np.zeros((nely, nelx, nelz))
        fixed_voxel[:, 0, :] = 1.0
        dlX_fixed = self.dlX[np.where(fixed_voxel.reshape([self.nele, 1]) == 1.0)[0], :]

        F = 0.1
        self.F_vector = torch.tensor([[F], [0.0], [0.0]], dtype=torch.float32).to(device)
        self.force_voxel = np.zeros((nely, nelx, nelz))
        self.force_voxel[yid, xid, zid] = 1
        dlX_force = self.dlX[np.where(
            self.force_voxel.reshape([self.nele, 1]) == 1)[0], :]

        self.dlX       = torch.tensor(self.dlX,    dtype=torch.float32, requires_grad=True).to(device)
        self.dlXSS     = torch.tensor(self.dlXSS,  dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_fixed = torch.tensor(dlX_fixed,   dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_force = torch.tensor(dlX_force,   dtype=torch.float32, requires_grad=True).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral kernel & networks
# ══════════════════════════════════════════════════════════════════════════════

_BAND_LOW, _BAND_HIGH = 0.0, 35.0


def _build_spectral_kernel(n_basis=100, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=None):
    rng = np.random.default_rng(seed)
    def _sample_1d(n):
        u = rng.uniform(0.0, 1.0, n)
        half = band_high - band_low
        val  = u * 2 * half
        return np.where(val < half, val - band_high, val - half + band_low)
    ky = _sample_1d(n_basis); kx = _sample_1d(n_basis); kz = _sample_1d(n_basis)
    return torch.tensor(np.stack((ky, kx, kz), axis=0), dtype=torch.float32)


class _SpectralBase(nn.Module):
    def __init__(self, trainable_kernel, n_basis=100,
                 band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=None):
        super().__init__()
        K = _build_spectral_kernel(n_basis, band_low, band_high, seed).to(device)
        if trainable_kernel:
            self.kernel1 = nn.Parameter(K)
        else:
            self.register_buffer('kernel1', K)
        self.register_buffer('_bias', torch.ones(1, K.shape[1], dtype=torch.float32).to(device))

    def _features(self, x):
        return torch.sin(torch.matmul(x, self.kernel1) + self._bias)


class Disp_Net(_SpectralBase):
    def __init__(self, n_basis=500, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=42):
        super().__init__(trainable_kernel=False,
                         n_basis=n_basis, band_low=band_low, band_high=band_high, seed=seed)

    def forward(self, x):
        return self._features(x)


import math

class TO_BlockNet(nn.Module):
    def __init__(self, n_cols=2, n_rows=2, n_depths=2, n_basis=50,
                 band_low=_BAND_LOW, band_high=_BAND_HIGH,
                 overlap=0.0, seed=None, device=device):
        super().__init__()
        self.n_cols   = int(n_cols);  self.n_rows   = int(n_rows)
        self.n_depths = int(n_depths)
        self.n_blocks = self.n_cols * self.n_rows * self.n_depths
        self.overlap  = overlap; self.device = device
        self.bias = nn.Parameter(
            torch.full((self.n_blocks,), math.log(0.3 / 0.7), device=device))
        K = _build_spectral_kernel(n_basis, band_low, band_high, seed).to(device)
        self.M = K.shape[1]
        self.kernels = nn.Parameter(K.unsqueeze(0).expand(self.n_blocks, -1, -1).clone())
        self.weights = nn.Parameter(torch.zeros(self.n_blocks, self.M, device=device))

    def _block_indices(self, coords):
        ys, xs, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        min_z, max_z = zs.min(), zs.max()
        row = ((ys-min_y)/(max_y-min_y+1e-9)*self.n_rows  ).long().clamp(0, self.n_rows-1)
        col = ((xs-min_x)/(max_x-min_x+1e-9)*self.n_cols  ).long().clamp(0, self.n_cols-1)
        dep = ((zs-min_z)/(max_z-min_z+1e-9)*self.n_depths).long().clamp(0, self.n_depths-1)
        return row * (self.n_cols * self.n_depths) + col * self.n_depths + dep

    def _soft_weights(self, coords):
        ys = coords[:, 0:1]; xs = coords[:, 1:2]; zs = coords[:, 2:3]
        min_y, max_y = ys.min().item(), ys.max().item()
        min_x, max_x = xs.min().item(), xs.max().item()
        min_z, max_z = zs.min().item(), zs.max().item()
        bh = (max_y-min_y)/self.n_rows; bw = (max_x-min_x)/self.n_cols
        bd = (max_z-min_z)/self.n_depths
        centers_y = torch.tensor(
            [min_y+(r+0.5)*bh for r in range(self.n_rows)
             for c in range(self.n_cols) for d in range(self.n_depths)],
            dtype=torch.float32, device=self.device)
        centers_x = torch.tensor(
            [min_x+(c+0.5)*bw for r in range(self.n_rows)
             for c in range(self.n_cols) for d in range(self.n_depths)],
            dtype=torch.float32, device=self.device)
        centers_z = torch.tensor(
            [min_z+(d+0.5)*bd for r in range(self.n_rows)
             for c in range(self.n_cols) for d in range(self.n_depths)],
            dtype=torch.float32, device=self.device)
        dy = (ys-centers_y.unsqueeze(0))/(bh*(0.5+self.overlap))
        dx = (xs-centers_x.unsqueeze(0))/(bw*(0.5+self.overlap))
        dz = (zs-centers_z.unsqueeze(0))/(bd*(0.5+self.overlap))
        return torch.softmax(-0.5*(dy**2+dx**2+dz**2), dim=1)

    def _all_blocks(self, coords):
        N, B, M = coords.shape[0], self.n_blocks, self.M
        K_flat = self.kernels.permute(1, 0, 2).reshape(3, B*M)
        z = (coords @ K_flat).view(N, B, M) + 1.0
        out = (torch.sin(z)*self.weights.unsqueeze(0)).sum(dim=2) + self.bias.unsqueeze(0)
        return torch.sigmoid(out)

    def forward(self, coords):
        coords  = coords.to(self.device)
        out_all = self._all_blocks(coords)
        if self.overlap <= 0:
            bidx = self._block_indices(coords)
            return out_all[torch.arange(coords.shape[0], device=self.device), bidx].unsqueeze(-1)
        else:
            w = self._soft_weights(coords)
            return (out_all * w).sum(dim=1, keepdim=True)

    def num_blocks(self):
        return self.n_blocks


# ══════════════════════════════════════════════════════════════════════════════
#  Analytical basis matrix, PDE loss, continuity loss
# ══════════════════════════════════════════════════════════════════════════════

def cal_matrix_3d(model, points, dlx_force, M):
    K = model.kernel1
    with torch.no_grad():
        z     = torch.matmul(points, K) + model._bias
        sin_z = torch.sin(z); cos_z = torch.cos(z)
        x1    = points[:, 1:2]
        sigma = torch.sigmoid(20.0 * (x1 + 0.5))
        bc    = 2.0 * (sigma - 0.5)
        d_bc  = 40.0 * sigma * (1.0 - sigma)
        u1    = sin_z * bc
        u_y   = cos_z * K[0:1, :] * bc
        u_x   = cos_z * K[1:2, :] * bc + sin_z * d_bc
        u_z   = cos_z * K[2:3, :] * bc
        v_force = model(dlx_force)
    return (u1.cpu().numpy(), u_x.cpu().numpy(),
            u_y.cpu().numpy(), u_z.cpu().numpy(), v_force.cpu().numpy())


_E_MOD, _NU_MOD = 1000.0, 0.3
_LAME_MU     = _E_MOD / (2.0*(1.0+_NU_MOD))
_LAME_LAMBDA = _E_MOD*_NU_MOD / ((1.0+_NU_MOD)*(1.0-2.0*_NU_MOD))


def pinnloss3d(weights1, u_x, u_y, u_z, v_force, problem, xPhys_m):
    xPhys_m = xPhys_m.reshape(-1, 1)
    dux_dx = (u_x @ weights1[:, 1]).reshape(-1, 1)
    dux_dy = (u_y @ weights1[:, 1]).reshape(-1, 1)
    dux_dz = (u_z @ weights1[:, 1]).reshape(-1, 1)
    duy_dx = (u_x @ weights1[:, 0]).reshape(-1, 1)
    duy_dy = (u_y @ weights1[:, 0]).reshape(-1, 1)
    duy_dz = (u_z @ weights1[:, 0]).reshape(-1, 1)
    duz_dx = (u_x @ weights1[:, 2]).reshape(-1, 1)
    duz_dy = (u_y @ weights1[:, 2]).reshape(-1, 1)
    duz_dz = (u_z @ weights1[:, 2]).reshape(-1, 1)
    eps11 = dux_dx; eps22 = duy_dy; eps33 = duz_dz
    eps12 = 0.5*(dux_dy+duy_dx); eps13 = 0.5*(dux_dz+duz_dx); eps23 = 0.5*(duy_dz+duz_dy)
    trace_sq = (eps11+eps22+eps33)**2
    diag_sq  = eps11**2+eps22**2+eps33**2
    energy   = (0.5*_LAME_LAMBDA*trace_sq + _LAME_MU*(diag_sq+2*eps12**2+2*eps13**2+2*eps23**2))
    energy   = energy * (xPhys_m**3.0)
    energy_ans = problem.V * torch.mean(energy)
    force_l    = torch.mean(0.1*(v_force @ weights1[:, 0]).reshape(-1, 1))
    return energy_ans - force_l, energy


def continueloss(to_model, problem):
    n_samples = 500; n_side = max(2, int(n_samples**0.5))
    coords_all = problem.dlX
    min_y, max_y = torch.min(coords_all[:,0]).item(), torch.max(coords_all[:,0]).item()
    min_x, max_x = torch.min(coords_all[:,1]).item(), torch.max(coords_all[:,1]).item()
    min_z, max_z = torch.min(coords_all[:,2]).item(), torch.max(coords_all[:,2]).item()
    losses = []; nc, nr, nd = to_model.n_cols, to_model.n_rows, to_model.n_depths
    def _b(r,c,d): return r*(nc*nd)+c*nd+d
    for k in range(nc-1):
        iface_x = min_x+(k+1)*(max_x-min_x)/nc
        ys = torch.linspace(min_y, max_y, n_side, device=to_model.device)
        zs = torch.linspace(min_z, max_z, n_side, device=to_model.device)
        y_flat,z_flat=(t.flatten().unsqueeze(1) for t in torch.meshgrid(ys,zs,indexing='ij'))
        pts = torch.cat([y_flat, torch.ones_like(y_flat)*iface_x, z_flat], dim=1)
        all_out = to_model._all_blocks(pts)
        row_idx = ((y_flat.squeeze()-min_y)/(max_y-min_y+1e-9)*nr).long().clamp(0,nr-1)
        dep_idx = ((z_flat.squeeze()-min_z)/(max_z-min_z+1e-9)*nd).long().clamp(0,nd-1)
        for r in range(nr):
            for d in range(nd):
                mask=(row_idx==r)&(dep_idx==d)
                if mask.sum()==0: continue
                losses.append(torch.mean((all_out[mask,_b(r,k,d)]-all_out[mask,_b(r,k+1,d)])**2))
    for k in range(nr-1):
        iface_y = min_y+(k+1)*(max_y-min_y)/nr
        xs = torch.linspace(min_x, max_x, n_side, device=to_model.device)
        zs = torch.linspace(min_z, max_z, n_side, device=to_model.device)
        x_flat,z_flat=(t.flatten().unsqueeze(1) for t in torch.meshgrid(xs,zs,indexing='ij'))
        pts = torch.cat([torch.ones_like(x_flat)*iface_y, x_flat, z_flat], dim=1)
        all_out = to_model._all_blocks(pts)
        col_idx = ((x_flat.squeeze()-min_x)/(max_x-min_x+1e-9)*nc).long().clamp(0,nc-1)
        dep_idx = ((z_flat.squeeze()-min_z)/(max_z-min_z+1e-9)*nd).long().clamp(0,nd-1)
        for c in range(nc):
            for d in range(nd):
                mask=(col_idx==c)&(dep_idx==d)
                if mask.sum()==0: continue
                losses.append(torch.mean((all_out[mask,_b(k,c,d)]-all_out[mask,_b(k+1,c,d)])**2))
    for k in range(nd-1):
        iface_z = min_z+(k+1)*(max_z-min_z)/nd
        ys = torch.linspace(min_y, max_y, n_side, device=to_model.device)
        xs = torch.linspace(min_x, max_x, n_side, device=to_model.device)
        y_flat,x_flat=(t.flatten().unsqueeze(1) for t in torch.meshgrid(ys,xs,indexing='ij'))
        pts = torch.cat([y_flat, x_flat, torch.ones_like(y_flat)*iface_z], dim=1)
        all_out = to_model._all_blocks(pts)
        row_idx = ((y_flat.squeeze()-min_y)/(max_y-min_y+1e-9)*nr).long().clamp(0,nr-1)
        col_idx = ((x_flat.squeeze()-min_x)/(max_x-min_x+1e-9)*nc).long().clamp(0,nc-1)
        for r in range(nr):
            for c in range(nc):
                mask=(row_idx==r)&(col_idx==c)
                if mask.sum()==0: continue
                losses.append(torch.mean((all_out[mask,_b(r,c,k)]-all_out[mask,_b(r,c,k+1)])**2))
    if not losses:
        return torch.tensor(0.0, device=to_model.device)
    return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  FEM compliance verification  (H8 hexahedral)
# ══════════════════════════════════════════════════════════════════════════════

def lk_H8(nu):
    A = np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],
                  [-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]])
    k = (1/144)*A.T@np.array([1, nu])
    K1=np.array([[k[0],k[1],k[1],k[2],k[4],k[4]],[k[1],k[0],k[1],k[3],k[5],k[6]],
                 [k[1],k[1],k[0],k[3],k[6],k[5]],[k[2],k[3],k[3],k[0],k[7],k[7]],
                 [k[4],k[5],k[6],k[7],k[0],k[1]],[k[4],k[6],k[5],k[7],k[1],k[0]]])
    K2=np.array([[k[8],k[7],k[11],k[5],k[3],k[6]],[k[7],k[8],k[11],k[4],k[2],k[4]],
                 [k[9],k[9],k[12],k[6],k[3],k[5]],[k[5],k[4],k[10],k[8],k[1],k[9]],
                 [k[3],k[2],k[4],k[1],k[8],k[11]],[k[10],k[3],k[5],k[11],k[9],k[12]]])
    K3=np.array([[k[5],k[6],k[3],k[8],k[11],k[7]],[k[6],k[5],k[3],k[9],k[12],k[9]],
                 [k[4],k[4],k[2],k[7],k[11],k[8]],[k[8],k[9],k[1],k[5],k[10],k[4]],
                 [k[11],k[12],k[9],k[10],k[5],k[3]],[k[1],k[11],k[8],k[3],k[4],k[2]]])
    K4=np.array([[k[13],k[10],k[10],k[12],k[9],k[9]],[k[10],k[13],k[10],k[11],k[8],k[7]],
                 [k[10],k[10],k[13],k[11],k[7],k[8]],[k[12],k[11],k[11],k[13],k[6],k[6]],
                 [k[9],k[8],k[7],k[6],k[13],k[10]],[k[9],k[7],k[8],k[6],k[10],k[13]]])
    K5=np.array([[k[0],k[1],k[7],k[2],k[4],k[3]],[k[1],k[0],k[7],k[3],k[5],k[10]],
                 [k[7],k[7],k[0],k[4],k[10],k[5]],[k[2],k[3],k[4],k[0],k[7],k[1]],
                 [k[4],k[5],k[10],k[7],k[0],k[7]],[k[3],k[10],k[5],k[1],k[7],k[0]]])
    K6=np.array([[k[13],k[10],k[6],k[12],k[9],k[11]],[k[10],k[13],k[6],k[11],k[8],k[1]],
                 [k[6],k[6],k[13],k[9],k[1],k[8]],[k[12],k[11],k[9],k[13],k[6],k[10]],
                 [k[9],k[8],k[1],k[6],k[13],k[6]],[k[11],k[1],k[8],k[10],k[6],k[13]]])
    KE = np.block([[K1,K2,K3,K4],[K2.T,K5,K6,K3.T],[K3.T,K6.T,K5.T,K2.T],[K4,K3,K2,K1.T]])
    return KE/((nu+1)*(1-2*nu))


def compute_compliance(xPhys, nelx, nely, nelz, E0=1000.0, Emin=1e-9, nu=0.3):
    nele = nelx*nely*nelz
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
    KE   = lk_H8(nu)
    j_idx, k_idx = np.meshgrid(np.arange(nely+1), np.arange(nelz+1), indexing='ij')
    fixednid = (k_idx*(nely+1)*(nelx+1)+j_idx).flatten()
    fixeddof = np.concatenate([3*fixednid, 3*fixednid+1, 3*fixednid+2])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddof)
    loadnid  = (nelz//2)*(nelx+1)*(nely+1) + nelx*(nely+1) + nely//2
    loaddof  = 3*loadnid+1
    F = sparse.lil_matrix((ndof, 1)); F[loaddof, 0] = -0.1; F = F.tocsr()
    nodegrd  = np.reshape(np.arange(1,(nely+1)*(nelx+1)+1),(nely+1,nelx+1),order='F')
    nodeids  = np.reshape(nodegrd[:-1,:-1],(nely*nelx,1),order='F')
    nodeidz  = np.arange(0, nelz*(nely+1)*(nelx+1), (nely+1)*(nelx+1))
    nodeids  = (np.tile(nodeids,(1,nelz))+np.tile(nodeidz,(nely*nelx,1))).flatten(order='F')
    edofVec  = 3*(nodeids-1)
    n_layer  = (nely+1)*(nelx+1)
    offset_node = np.array([0,nely+1,nely+2,1,n_layer,n_layer+nely+1,n_layer+nely+2,n_layer+1])
    edofMat  = np.zeros((nele, 24), dtype=int)
    for i in range(24):
        edofMat[:, i] = edofVec + 3*offset_node[i//3] + (i%3)
    iK = np.repeat(edofMat, 24, axis=1).flatten()
    jK = np.repeat(edofMat, 24, axis=0).flatten()
    xf = xPhys.flatten(order='F')
    sK = (KE.ravel()[np.newaxis,:] * (Emin+xf**1*(E0-Emin))[:,np.newaxis]).ravel()
    K  = sparse.coo_matrix((sK,(iK,jK)), shape=(ndof,ndof)).tocsr()
    K  = (K+K.T)*0.5
    U  = np.zeros(ndof)
    U[freedofs] = spsolve(K[np.ix_(freedofs,freedofs)], F[freedofs].toarray().ravel())
    Ue = U[edofMat]
    ce = np.sum((Ue@KE)*Ue, axis=1)
    return float(np.sum((Emin+xf*(E0-Emin))*ce))


# ══════════════════════════════════════════════════════════════════════════════
#  Binarization + connectivity filter
# ══════════════════════════════════════════════════════════════════════════════

def binarize_3d(tt, threshold=0.4):
    """Binary + keep largest connected component."""
    tt_bin = (tt >= threshold).astype(np.uint8)
    labeled, n_feat = label(tt_bin)
    if n_feat == 0:
        return tt_bin
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Main solver  — λ_pf 作为可配置参数  (mirrors 2-D RFM_TONN)
# ══════════════════════════════════════════════════════════════════════════════

class RFM_TONN():
    def __init__(self, problem, to_model, disp_model, lambda_pf=0.1):
        self.problem      = problem
        self.disp_model   = disp_model
        self.to_model     = to_model
        self.lambda_pf    = lambda_pf          # ← 敏感度分析目标参数
        self.total_epoch  = 0

        self.to_optimizer = optim.Adam(self.to_model.parameters(), lr=0.03)
        self.to_scheduler = optim.lr_scheduler.LinearLR(
            self.to_optimizer, start_factor=1.0,
            end_factor=1e-2/3e-2, total_iters=400)

        self.coord = problem.dlX_disp()
        M = disp_model.kernel1.shape[1]
        _, u_x, u_y, u_z, v_force = cal_matrix_3d(
            self.disp_model, self.coord, self.problem.dlX_force, M)
        self.u_x     = torch.tensor(u_x,     dtype=torch.float32, device=device)
        self.u_y     = torch.tensor(u_y,     dtype=torch.float32, device=device)
        self.u_z     = torch.tensor(u_z,     dtype=torch.float32, device=device)
        self.v_force = torch.tensor(v_force, dtype=torch.float32, device=device)

    def fit_disp_init(self):
        M = self.disp_model.kernel1.shape[1]
        self.weights1 = nn.Parameter(
            torch.zeros(M, 3, dtype=torch.float32, device=device))
        self.disp_optimizer = optim.Adam([self.weights1], lr=5e-6)
        xPhys_init = torch.ones(self.coord.shape[0], 1, device=device) * 0.3
        for _ in range(1000):
            loss, _ = pinnloss3d(self.weights1, self.u_x, self.u_y, self.u_z,
                                 self.v_force, self.problem, xPhys_init)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

        _, u_x, u_y, u_z, v_force = cal_matrix_3d(
            self.disp_model, self.problem.dlX, self.problem.dlX_force, M)
        u_x     = torch.tensor(u_x,     dtype=torch.float32, device=device)
        u_y     = torch.tensor(u_y,     dtype=torch.float32, device=device)
        u_z     = torch.tensor(u_z,     dtype=torch.float32, device=device)
        v_force = torch.tensor(v_force, dtype=torch.float32, device=device)
        xPhys   = torch.ones(self.problem.dlX.shape[0], 1, device=device) * 0.3
        _, energy_c = pinnloss3d(self.weights1, u_x, u_y, u_z, v_force, self.problem, xPhys)
        self.c1  = energy_c
        self.c_0 = torch.mean(energy_c)

    def to_loss(self, coord):
        self.total_epoch += 1
        xPhys_m = self.to_model(coord)
        alpha   = min(self.problem.alpha_init + self.problem.alpha_delta*self.total_epoch,
                      self.problem.alpha_max)
        _, energy_c = pinnloss3d(self.weights1, self.u_x, self.u_y, self.u_z,
                                 self.v_force, self.problem, xPhys_m)

        class ComputeDeDrho(torch.autograd.Function):
            @staticmethod
            def forward(ctx, xPhys_m, energy_c, coord):
                ctx.save_for_backward(xPhys_m, energy_c, coord); return energy_c
            @staticmethod
            def backward(ctx, denergy):
                xPhys_m, energy_c, coord = ctx.saved_tensors
                grad_energy = torch.autograd.grad(
                    outputs=energy_c, inputs=xPhys_m, grad_outputs=denergy,
                    create_graph=True, retain_graph=True)[0]
                return -grad_energy, torch.zeros_like(energy_c), torch.zeros_like(coord)

        c           = torch.mean(ComputeDeDrho.apply(xPhys_m, energy_c, coord))
        xPhys_dlX   = self.to_model(self.problem.dlX)
        vf          = torch.mean(xPhys_dlX)

        epsilon     = 8e-3; gamma = 0.1
        grad_phi    = torch.autograd.grad(
            outputs=xPhys_m.sum(), inputs=coord, create_graph=True, retain_graph=True)[0]
        grad_term   = 0.5*epsilon*torch.mean(torch.sum(grad_phi**2, dim=1))
        double_well = (1/epsilon)*torch.mean(xPhys_m**2*(1-xPhys_m)**2)
        reg_term    = gamma*(0.01*grad_term + double_well)

        # ── λ_pf 控制正则化强度  (与2D完全对称) ─────────────────────────
        loss = (alpha*(vf/self.problem.volfrac-1.0)**2
                + c/self.c_0.detach()
                + self.lambda_pf*reg_term)

        cur_lr = self.to_optimizer.param_groups[0]['lr']
        print(f'  Epoch {self.total_epoch:>3d} | loss {loss.item():.5f} | '
              f'c {c.item():.5f} | LR {cur_lr:.2e}', end='\r')
        return loss

    def fit_disp(self, epochs=50):
        for _ in range(epochs):
            xPhys_m = self.to_model(self.coord)
            loss, _ = pinnloss3d(self.weights1, self.u_x, self.u_y, self.u_z,
                                 self.v_force, self.problem, xPhys_m)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

    def fit_to(self, epochs=400):
        for epoch in range(epochs):
            self.fit_disp(50)
            loss = self.to_loss(self.coord)
            self.to_optimizer.zero_grad(); loss.backward()
            self.to_optimizer.step(); self.to_scheduler.step()
        print()   # newline after \r


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_iso_save(xPhys, tag, title=''):
    """Marching-cubes isosurface render → PDF."""
    nely, nelx, nelz = xPhys.shape
    nelm    = max(nely, nelx, nelz)
    padding = np.zeros([nely+2, nelx+2, nelz+2])
    padding[1:-1, 1:-1, 1:-1] = xPhys
    try:
        verts, faces, normals, values = measure.marching_cubes(padding, 0.5)
    except Exception:
        print(f'  [skip iso: no surface at threshold 0.5 for {tag}]')
        return
    fig = plt.figure(figsize=(16, 9))
    ax  = fig.add_subplot(1, 1, 1, projection='3d')
    ls  = LightSource(azdeg=45, altdeg=-45)
    f_coord = np.take(verts, faces, axis=0)
    f_norm  = np.cross(f_coord[:,2]-f_coord[:,0], f_coord[:,1]-f_coord[:,0])
    cl      = ls.shade_normals(f_norm)
    norm    = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper  = cm.ScalarMappable(norm=norm, cmap=cm.gray_r)
    rgb     = mapper.to_rgba(cl).reshape([-1, 4])
    mesh    = Poly3DCollection(
        np.take(verts, faces, axis=0)/nelx*np.array([[nelm/nely, nelm/nelm, nelm/nelz]]))
    ax.add_collection3d(mesh); mesh.set_facecolors(rgb)
    ax.view_init(-150, -120, vertical_axis='x')
    ax.set_box_aspect(aspect=(nelx, nelz, nely))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fname = f'sensitivity3d_{tag}.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


# ══════════════════════════════════════════════════════════════════════════════
#  Problem / shared model setup
# ══════════════════════════════════════════════════════════════════════════════

nelx = 60; nely = 20; nelz = 8
xid  = 59; yid  = 9;  zid  = 3
vf   = 0.3

# Displacement network is shared across all runs (fixed kernel)
disp_model_h = Disp_Net(n_basis=5000, band_low=0.0, band_high=35.0, seed=42).to(device)
M_disp       = disp_model_h.kernel1.shape[1]
print(f'Disp_Net basis M = {M_disp}')

# ══════════════════════════════════════════════════════════════════════════════
#  Sensitivity sweep:  λ_pf ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

records = []   # dict per case

for lam in LAMBDA_VALUES:
    print(f"\n{'='*60}")
    print(f"  λ_pf = {lam}")
    print(f"{'='*60}")

    problem  = Cantilever_Beam_3D(nelx, nely, nelz, xid, yid, zid, vf)
    to_model = TO_BlockNet(
        n_cols=4, n_rows=4, n_depths=4,
        n_basis=30, band_low=0.0, band_high=35.0,
        overlap=0.2,
    ).to(device)

    opt = RFM_TONN(problem, to_model, disp_model_h, lambda_pf=lam)
    opt.fit_disp_init()
    opt.fit_to(400)

    # ── 超采样连续密度场 (2*nely × 2*nelx × 2*nelz) ─────────────────
    with torch.no_grad():
        xPhys_ss = opt.to_model(problem.dlX.to(device))
    tt_cont = xPhys_ss.cpu().numpy().reshape(nely, nelx, nelz)

    # ── 二值化 + FEM compliance ──────────────────────────────────────
    tt_bin = binarize_3d(tt_cont, threshold=0.4)

    # FEM 在超采样分辨率 (2*nely, 2*nelx, 2*nelz) 上计算
    c_cont = compute_compliance(tt_cont,              nelx, nely, nelz)
    c_bin  = compute_compliance(tt_bin.astype(float), nelx, nely, nelz)

    # 二值化程度
    flat         = tt_cont.flatten()
    binary_ratio = float(np.mean((flat <= 0.2) | (flat >= 0.8)))

    print(f"\n  → Continuous compliance : {c_cont:.8f}")
    print(f"  → Binarized  compliance : {c_bin:.8f}")
    print(f"  → Binary ratio          : {binary_ratio:.4%}")

    records.append(dict(
        lam         = lam,
        tt_cont     = tt_cont,
        tt_bin      = tt_bin,
        c_cont      = c_cont,
        c_bin       = c_bin,
        bin_ratio   = binary_ratio,
    ))

    # ── 保存等值面图 ──────────────────────────────────────────────────
    tag_str = str(lam).replace('.', 'p')
    plot_iso_save(
        tt_cont, f'lam{tag_str}_continuous',
        title=rf'Continuous  |  $\lambda_{{pf}}$ = {lam}  |  c = {c_cont:.6f}')
    plot_iso_save(
        tt_bin.astype(float), f'lam{tag_str}_binarized',
        title=rf'Binarized   |  $\lambda_{{pf}}$ = {lam}  |  c = {c_bin:.6f}')

    del opt, to_model, problem
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  数值汇总表  (与2D版本格式完全一致)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(f"  {'lambda_pf':<10}  {'c (continuous)':>22}  {'c (binarized)':>22}  {'binary ratio':>14}")
print("-"*80)
for rec in records:
    print(f"  {rec['lam']:<10.3f}  {rec['c_cont']:>22.8f}  {rec['c_bin']:>22.8f}  {rec['bin_ratio']:>13.4%}")
print("="*80)


# ══════════════════════════════════════════════════════════════════════════════
#  汇总对比图  (compliance vs λ_pf + binary ratio vs λ_pf)
# ══════════════════════════════════════════════════════════════════════════════

lams    = [r['lam']      for r in records]
c_conts = [r['c_cont']   for r in records]
c_bins  = [r['c_bin']    for r in records]
b_ratios= [r['bin_ratio'] for r in records]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(lams, c_conts, 'o-', color='#3498DB', linewidth=2, label='Continuous')
ax.plot(lams, c_bins,  's--', color='#E74C3C', linewidth=2, label='Binarized')
ax.set_xscale('log'); ax.set_xlabel(r'$\lambda_{\mathrm{pf}}$', fontsize=12)
ax.set_ylabel('FEM Compliance', fontsize=12)
ax.set_title('Compliance vs regularisation weight (3D)', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.plot(lams, [100*b for b in b_ratios], 'D-', color='#2ECC71', linewidth=2)
ax.set_xscale('log'); ax.set_xlabel(r'$\lambda_{\mathrm{pf}}$', fontsize=12)
ax.set_ylabel('Binary ratio (%)', fontsize=12)
ax.set_title('Binarization degree vs regularisation weight (3D)', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('sensitivity3d_summary.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('✓ sensitivity3d_summary.pdf saved')