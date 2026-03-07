import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from IPython import display
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.sparse as sp
from scipy.ndimage import label
import time

device = 'cuda:2'

# ══════════════════════════════════════════════════════════════════════════════
#  Problem definitions
# ══════════════════════════════════════════════════════════════════════════════

class Problems():
    def dlX_disp(self):
        domain_xcoord = np.random.uniform(
            -self.nelx / (2 * self.nelm),
             self.nelx / (2 * self.nelm),
            (self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0], 1)
        )
        domain_ycoord = np.random.uniform(
            -self.nely / (2 * self.nelm),
             self.nely / (2 * self.nelm),
            (self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0], 1)
        )
        domain_coord = np.concatenate((domain_ycoord, domain_xcoord), axis=1)
        coord = np.concatenate(
            (self.dlX_fixed.cpu().detach().numpy(),
             self.dlX_force.cpu().detach().numpy()), axis=0
        )
        coord = np.concatenate((coord, domain_coord), axis=0)
        coord = torch.tensor(coord, dtype=torch.float32, requires_grad=True).to(device)
        return coord


class Cantilever_Beam_2D(Problems):
    def __init__(self, nelx, nely, xid, yid, vf):
        self.xid  = xid
        self.yid  = yid
        self.nelx = nelx
        self.nely = nely
        self.nele    = self.nelx * self.nely
        self.nelm    = max(self.nelx, self.nely)
        self.volfrac = vf
        self.E0      = 1
        self.nu      = 0.3
        self.batch_size  = 25000
        self.alpha_init  = 1
        self.alpha_max   = 100
        self.alpha_delta = 0.5
        self.penal       = 3.0

        c_y, c_x = np.meshgrid(
            np.linspace(-(self.nely)/(2*self.nelm), (self.nely)/(2*self.nelm), self.nely),
            np.linspace(-(self.nelx)/(2*self.nelm), (self.nelx)/(2*self.nelm), self.nelx),
            indexing='ij')
        self.dlX = np.stack((c_y.reshape([-1]), c_x.reshape([-1])), axis=1).reshape([-1, 2])

        c_y, c_x = np.meshgrid(
            np.linspace(-(self.nely)/(2*self.nelm), (self.nely)/(2*self.nelm), 2*self.nely),
            np.linspace(-(self.nelx)/(2*self.nelm), (self.nelx)/(2*self.nelm), 2*self.nelx),
            indexing='ij')
        self.dlXSS = np.stack((c_y.reshape([-1]), c_x.reshape([-1])), axis=1).reshape([-1, 2])
        self.V = (
            (np.max(self.dlX[:, 0]) - np.min(self.dlX[:, 0])) *
            (np.max(self.dlX[:, 1]) - np.min(self.dlX[:, 1]))
        )

        fixed_voxel       = np.zeros((self.nely, self.nelx))
        fixed_voxel[:, 0] = 1.0
        fixed_voxel       = fixed_voxel.reshape([self.nele, 1])
        dlX_fixed         = self.dlX[np.where(fixed_voxel == 1.0)[0], :]

        F = 0.1
        self.F_vector              = torch.tensor([[F], [0.0]], dtype=torch.float32).to(device)
        self.force_voxel           = np.zeros((self.nely, self.nelx))
        self.force_voxel[yid, xid] = 1
        force_voxel = self.force_voxel.reshape([self.nele, 1])
        dlX_force   = self.dlX[np.where(force_voxel == 1)[0], :]

        self.dlX       = torch.tensor(self.dlX,    dtype=torch.float32, requires_grad=True).to(device)
        self.dlXSS     = torch.tensor(self.dlXSS,  dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_fixed = torch.tensor(dlX_fixed,   dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_force = torch.tensor(dlX_force,   dtype=torch.float32, requires_grad=True).to(device)

    def analytical_fixed_BC(self, u, coord):
        u = u * 2 * (1 / (1 + torch.exp(-20 * (coord[:, 1:2] + 0.5))) - 0.5)
        return u


_BAND_LOW, _BAND_HIGH = 0.0, 10.0


def _build_spectral_kernel(n_basis=100, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=None):
    rng = np.random.default_rng(seed)

    def _sample_1d(n):
        u    = rng.uniform(0.0, 1.0, n)
        half = band_high - band_low
        val  = u * 2 * half
        return np.where(val < half, val - band_high, val - half + band_low)

    ky = _sample_1d(n_basis)
    kx = _sample_1d(n_basis)
    return torch.tensor(np.stack((ky, kx), axis=0), dtype=torch.float32)


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
        return torch.sin(x @ self.kernel1 + self._bias)


class Disp_Net(_SpectralBase):
    def __init__(self, n_basis=400, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=42):
        super().__init__(trainable_kernel=False,
                         n_basis=n_basis, band_low=band_low, band_high=band_high, seed=seed)

    def forward(self, x):
        return self._features(x)


class TO_BlockNet(nn.Module):
    def __init__(self, n_cols=2, n_rows=2, n_basis=50,
                 band_low=_BAND_LOW, band_high=_BAND_HIGH,
                 overlap=0.0, seed=None, device='cuda:2'):
        super().__init__()
        self.n_cols   = int(n_cols)
        self.n_rows   = int(n_rows)
        self.n_blocks = self.n_cols * self.n_rows
        self.overlap  = overlap
        self.device   = device

        K = _build_spectral_kernel(n_basis, band_low, band_high, seed).to(device)
        self.M       = K.shape[1]
        self.kernels = nn.Parameter(K.unsqueeze(0).expand(self.n_blocks, -1, -1).clone())
        self.weights = nn.Parameter(torch.zeros(self.n_blocks, self.M, device=device))

    def _block_indices(self, coords):
        ys, xs       = coords[:, 0], coords[:, 1]
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        col = ((xs-min_x)/(max_x-min_x+1e-9)*self.n_cols).long().clamp(0, self.n_cols-1)
        row = ((ys-min_y)/(max_y-min_y+1e-9)*self.n_rows).long().clamp(0, self.n_rows-1)
        return row * self.n_cols + col

    def _soft_weights(self, coords):
        ys, xs       = coords[:, 0:1], coords[:, 1:2]
        min_y, max_y = ys.min().item(), ys.max().item()
        min_x, max_x = xs.min().item(), xs.max().item()
        bh = (max_y - min_y) / self.n_rows
        bw = (max_x - min_x) / self.n_cols
        centers_y = torch.tensor(
            [min_y+(r+0.5)*bh for r in range(self.n_rows) for c in range(self.n_cols)],
            dtype=torch.float32, device=self.device)
        centers_x = torch.tensor(
            [min_x+(c+0.5)*bw for r in range(self.n_rows) for c in range(self.n_cols)],
            dtype=torch.float32, device=self.device)
        dy = (ys - centers_y.unsqueeze(0)) / (bh * (0.5 + self.overlap))
        dx = (xs - centers_x.unsqueeze(0)) / (bw * (0.5 + self.overlap))
        return torch.softmax(-0.5 * (dy**2 + dx**2), dim=1)

    def _all_blocks(self, coords):
        N, B, M = coords.shape[0], self.n_blocks, self.M
        K_flat  = self.kernels.permute(1, 0, 2).reshape(2, B * M)
        z       = (coords @ K_flat).view(N, B, M) + 1.0
        out     = (torch.sin(z) * self.weights.unsqueeze(0)).sum(dim=2)
        return torch.sigmoid(out)

    def forward(self, coords):
        coords  = coords.to(self.device)
        out_all = self._all_blocks(coords)
        if self.overlap <= 0:
            bidx = self._block_indices(coords)
            N    = coords.shape[0]
            return out_all[torch.arange(N, device=self.device), bidx].unsqueeze(-1)
        else:
            w = self._soft_weights(coords)
            return (out_all * w).sum(dim=1, keepdim=True)

    def num_blocks(self):
        return self.n_blocks


def cal_matrix(model, points, dlx_force, M_size):
    K = model.kernel1
    with torch.no_grad():
        z     = points @ K + model._bias
        sin_z = torch.sin(z)
        cos_z = torch.cos(z)
        x1    = points[:, 1:2]
        sigma = torch.sigmoid(20.0 * (x1 + 0.5))
        bc    = 2.0 * (sigma - 0.5)
        d_bc  = 40.0 * sigma * (1.0 - sigma)
        u1    = sin_z * bc
        u_y   = cos_z * K[0:1, :] * bc
        u_x   = cos_z * K[1:2, :] * bc + sin_z * d_bc
        v_force = model(dlx_force)
    return u1.cpu().numpy(), u_x.cpu().numpy(), u_y.cpu().numpy(), v_force.cpu().numpy()


_E_MOD, _NU_MOD = 1000.0, 0.3
_LAME_MU        = _E_MOD / (2.0 * (1.0 + _NU_MOD))
_LAME_LAMBDA    = _E_MOD * _NU_MOD / (1.0 - _NU_MOD**2)


def pinnloss2(weights1, u_x, u_y, v_force, problem, xPhys_m):
    xPhys_m  = xPhys_m.reshape(-1, 1)
    ux = (u_x @ weights1[:, 1]).reshape(-1, 1)
    uy = (u_y @ weights1[:, 1]).reshape(-1, 1)
    vx = (u_x @ weights1[:, 0]).reshape(-1, 1)
    vy = (u_y @ weights1[:, 0]).reshape(-1, 1)
    eps11    = ux
    eps12    = 0.5 * (uy + vx)
    eps22    = vy
    trace_sq = (eps11 + eps22) ** 2
    diag_sq  = eps11 * eps11 + eps22 * eps22
    energy   = (0.5*_LAME_LAMBDA*trace_sq + _LAME_MU*(diag_sq + 2.0*eps12*eps12))
    energy   = energy * (xPhys_m ** 3.0)
    energy_ans = problem.V * torch.mean(energy)
    force_l    = torch.mean(0.1 * (v_force @ weights1[:, 0]).reshape(-1, 1))
    loss       = energy_ans - force_l
    return loss, energy


def continueloss(to_model, problem):
    n_samples = 500
    min_y = torch.min(problem.dlX[:, 0]).item()
    max_y = torch.max(problem.dlX[:, 0]).item()
    min_x = torch.min(problem.dlX[:, 1]).item()
    max_x = torch.max(problem.dlX[:, 1]).item()
    losses = []
    for k in range(to_model.n_cols - 1):
        iface_x = min_x + (k+1)*(max_x-min_x)/to_model.n_cols
        ys  = torch.linspace(min_y, max_y, n_samples, device=to_model.device).unsqueeze(1)
        pts = torch.cat([ys, torch.full_like(ys, iface_x)], dim=1)
        all_out = to_model._all_blocks(pts)
        row_idx = ((ys.squeeze(1)-min_y)/(max_y-min_y+1e-9)*to_model.n_rows).long().clamp(0, to_model.n_rows-1)
        for r in range(to_model.n_rows):
            mask = (row_idx == r)
            if mask.sum() == 0: continue
            losses.append(torch.mean((all_out[mask, r*to_model.n_cols+k] - all_out[mask, r*to_model.n_cols+(k+1)])**2))
    for k in range(to_model.n_rows - 1):
        iface_y = min_y + (k+1)*(max_y-min_y)/to_model.n_rows
        xs  = torch.linspace(min_x, max_x, n_samples, device=to_model.device).unsqueeze(1)
        pts = torch.cat([torch.full_like(xs, iface_y), xs], dim=1)
        all_out = to_model._all_blocks(pts)
        col_idx = ((xs.squeeze(1)-min_x)/(max_x-min_x+1e-9)*to_model.n_cols).long().clamp(0, to_model.n_cols-1)
        for c in range(to_model.n_cols):
            mask = (col_idx == c)
            if mask.sum() == 0: continue
            losses.append(torch.mean((all_out[mask, k*to_model.n_cols+c] - all_out[mask, (k+1)*to_model.n_cols+c])**2))
    if not losses:
        return torch.tensor(0.0, device=to_model.device)
    return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  RFM_TONN — λ_pf 作为可配置参数
# ══════════════════════════════════════════════════════════════════════════════

class RFM_TONN():
    def __init__(self, problem, to_model, disp_model, lambda_pf=0.5):
        self.problem    = problem
        self.disp_model = disp_model
        self.to_model   = to_model
        self.lambda_pf  = lambda_pf          # ← 敏感度分析的目标参数
        self.total_epoch = 0
        self.to_optimizer = optim.Adam(self.to_model.parameters(), lr=0.01)
        self.to_scheduler = optim.lr_scheduler.LinearLR(
            self.to_optimizer, start_factor=1.0,
            end_factor=5e-3/1e-2, total_iters=400)
        self.coord = problem.dlX_disp()

        _, u_x, u_y, v_force = cal_matrix(
            self.disp_model, self.coord, self.problem.dlX_force, M)
        self.u_x     = torch.tensor(u_x,     dtype=torch.float32, device=device)
        self.u_y     = torch.tensor(u_y,     dtype=torch.float32, device=device)
        self.v_force = torch.tensor(v_force, dtype=torch.float32, device=device)

    def fit_disp_init(self):
        self.weights1 = nn.Parameter(
            torch.zeros(self.disp_model.kernel1.shape[1], 2,
                        dtype=torch.float32, device=device))
        self.disp_optimizer = optim.Adam([self.weights1], lr=1e-6)
        xPhys_init = torch.full((self.coord.shape[0], 1), 0.5, device=device)
        for _ in range(1000):
            loss, _ = pinnloss2(self.weights1, self.u_x, self.u_y,
                                self.v_force, self.problem, xPhys_init)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

        _, u_x, u_y, v_force = cal_matrix(
            self.disp_model, self.problem.dlX, self.problem.dlX_force, M)
        u_x     = torch.tensor(u_x,     dtype=torch.float32, device=device)
        u_y     = torch.tensor(u_y,     dtype=torch.float32, device=device)
        v_force = torch.tensor(v_force, dtype=torch.float32, device=device)
        xPhys_ref = torch.full((self.problem.dlX.shape[0], 1), 0.5, device=device)
        _, energy_c = pinnloss2(self.weights1, u_x, u_y, v_force, self.problem, xPhys_ref)
        self.c1  = energy_c
        self.c_0 = torch.mean(energy_c)

    def to_loss(self, coord):
        self.total_epoch += 1
        xPhys_m = self.to_model(coord)
        alpha   = min(self.problem.alpha_init + self.problem.alpha_delta * self.total_epoch,
                      self.problem.alpha_max)

        _, energy_c = pinnloss2(self.weights1, self.u_x, self.u_y,
                                self.v_force, self.problem, xPhys_m)

        class ComputeDeDrho(torch.autograd.Function):
            @staticmethod
            def forward(ctx, xPhys_m, energy_c, coord):
                ctx.save_for_backward(xPhys_m, energy_c, coord)
                return energy_c
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

        epsilon     = 8e-3
        gamma       = 0.1
        grad_phi    = torch.autograd.grad(
            outputs=xPhys_m.sum(), inputs=coord,
            create_graph=True, retain_graph=True)[0]
        grad_term   = 0.5 * epsilon * torch.mean(torch.sum(grad_phi**2, dim=1))
        double_well = (1/epsilon) * torch.mean(xPhys_m**2 * (1-xPhys_m)**2)
        reg_term    = gamma * (0.01*grad_term + double_well)

        # ── λ_pf 控制正则化强度 ──────────────────────────────────────────
        loss = (alpha * (vf / self.problem.volfrac - 1.0)**2
                + c / self.c_0.detach()
                + self.lambda_pf * reg_term)

        cur_lr = self.to_optimizer.param_groups[0]['lr']
        print(f'  Epoch {self.total_epoch:>3d} | loss {loss.item():.5f} | '
              f'c {c.item():.5f} | LR {cur_lr:.2e}', end='\r')
        return loss

    def fit_disp(self, epochs=50):
        for _ in range(epochs):
            xPhys_m = self.to_model(self.coord)
            loss, _ = pinnloss2(self.weights1, self.u_x, self.u_y,
                                self.v_force, self.problem, xPhys_m)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

    def fit_to(self, epochs=400):
        for epoch in range(epochs):
            self.fit_disp(50)
            loss = self.to_loss(self.coord)
            self.to_optimizer.zero_grad()
            loss.backward()
            self.to_optimizer.step()
            self.to_scheduler.step()
        print()   # newline after \r


# ══════════════════════════════════════════════════════════════════════════════
#  Compliance calculator (FEM)
# ══════════════════════════════════════════════════════════════════════════════

def calc_compliance(xPhys, penal=3):
    nely, nelx = xPhys.shape
    E0   = 1000.0
    Emin = 1e-9
    nu   = 0.3

    A11 = np.array([[ 12,  3, -6, -3],[  3, 12,  3,  0],[ -6,  3, 12, -3],[ -3,  0, -3, 12]])
    A12 = np.array([[ -6, -3,  0,  3],[ -3, -6, -3, -6],[  0, -3, -6,  3],[  3, -6,  3, -6]])
    B11 = np.array([[ -4,  3, -2,  9],[  3, -4, -9,  4],[ -2, -9, -4, -3],[  9,  4, -3, -4]])
    B12 = np.array([[  2, -3,  4, -9],[ -3,  2,  9, -2],[  4,  9,  2,  3],[ -9, -2,  3,  2]])
    KE  = (1/(1-nu**2)/24) * (
        np.block([[A11,A12],[A12.T,A11]]) + nu*np.block([[B11,B12],[B12.T,B11]]))

    ndof    = 2*(nelx+1)*(nely+1)
    nodenrs = np.arange(1,(1+nelx)*(1+nely)+1).reshape(1+nely,1+nelx,order='F')
    edofVec = (2*nodenrs[:nely,:nelx]+1).reshape(nelx*nely,1,order='F')
    offsets = np.array([0,1,2*nely+2,2*nely+3,2*nely,2*nely+1,-2,-1])
    edofMat = (np.tile(edofVec,(1,8))+np.tile(offsets,(nelx*nely,1))-1).astype(int)
    iK = np.tile(edofMat,8).flatten().astype(int)
    jK = np.repeat(edofMat,8,axis=1).flatten().astype(int)

    F_vec = np.zeros(ndof)
    F_vec[2*(nelx*(nely+1)+nely//2)+1] = -0.1
    fixeddofs = np.arange(0, 2*(nely+1))
    freedofs  = np.setdiff1d(np.arange(ndof), fixeddofs)

    elem_mods = Emin + xPhys.flatten(order='F')**penal * (E0-Emin)
    sK = (KE.flatten(order='F')[:,None] * elem_mods[None,:]).flatten(order='F')
    K  = sp.csr_matrix((sK,(iK,jK)), shape=(ndof,ndof))
    K  = (K + K.T) / 2
    U  = np.zeros(ndof)
    U[freedofs] = spsolve(K[freedofs,:][:,freedofs], F_vec[freedofs])
    Ue = U[edofMat]
    ce = (Ue @ KE * Ue).sum(axis=1).reshape(nely,nelx,order='F')
    return float(np.sum((Emin + xPhys**penal*(E0-Emin)) * ce))


def binarize(tt, threshold=0.4):
    tt_bin  = (tt >= threshold).astype(np.uint8)
    labeled, n_feat = label(tt_bin, structure=np.ones((3,3),dtype=int))
    if n_feat == 0:
        return tt_bin
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Problem / model setup
# ══════════════════════════════════════════════════════════════════════════════

nelx = 60
nely = 20
xid  = 59
yid  = 9
vf   = 0.5

disp_model_h = Disp_Net(n_basis=1200, band_low=0.0, band_high=40.0, seed=42).to(device)
M            = disp_model_h.kernel1.shape[1]

# ══════════════════════════════════════════════════════════════════════════════
#  Sensitivity sweep:  λ_pf ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1,0.2]
N_CASES       = len(LAMBDA_VALUES)

records = []   # 每个元素: dict(lam, tt_cont, tt_bin, c_cont, c_bin)

for lam in LAMBDA_VALUES:
    print(f"\n{'='*60}")
    print(f"  λ_pf = {lam}")
    print(f"{'='*60}")

    problem  = Cantilever_Beam_2D(nelx, nely, xid, yid, vf)
    to_model = TO_BlockNet(n_cols=4, n_rows=4, n_basis=50,
                           band_low=5.0, band_high=40.0,
                           overlap=0.2).to(device)

    opt = RFM_TONN(problem, to_model, disp_model_h, lambda_pf=lam)
    opt.fit_disp_init()
    opt.fit_to(400)

    # ── 连续密度场 ──────────────────────────────────────────────────────
    with torch.no_grad():
        xPhys_ss = opt.to_model(problem.dlXSS.to(device))
    tt_cont = xPhys_ss.cpu().numpy().reshape(2*nely, 2*nelx)

    # ── 直接用全分辨率 (2*nely × 2*nelx) 计算 compliance ───────────────
    tt_bin = binarize(tt_cont)

    c_cont = calc_compliance(tt_cont)
    c_bin  = calc_compliance(tt_bin.astype(float))
    # ── 二值化程度占比 ──────────────────────────────────────────────────
    flat = tt_cont.flatten()
    binary_ratio = float(np.mean((flat <= 0.3) | (flat >= 0.7)))

    print(f"  → Continuous compliance : {c_cont:.8f}")
    print(f"  → Binarized  compliance : {c_bin:.8f}")

    records.append(dict(
        lam    = lam,
        tt_cont = tt_cont,
        tt_bin  = tt_bin,
        c_cont  = c_cont,
        c_bin   = c_bin,
        bin_ratio  = binary_ratio, 
    ))

    # 释放显存
    del opt, to_model, problem
    torch.cuda.empty_cache()



# ══════════════════════════════════════════════════════════════════════════════
#  Save each case as individual files
# ══════════════════════════════════════════════════════════════════════════════

BG = '#F8F7F4'

for rec in records:
    lam    = rec['lam']
    c_cont = rec['c_cont']
    c_bin  = rec['c_bin']
    tag    = str(lam).replace('.', 'p')   # e.g. 0.01 → "0p01"

    # ── 连续密度场 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(-rec['tt_cont'], cmap='gray', aspect='equal')
    ax.axis('off')
    plt.tight_layout()
    fname = f'sensitivity_lam{tag}_continuous.pdf'
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')

    # ── 二值化密度场 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    ax.imshow(-rec['tt_bin'].astype(float), cmap='gray', aspect='equal')
    ax.axis('off')
    ax.set_title(
        rf'Binarized  |  $\lambda_{{\mathrm{{pf}}}}$ = {lam}  |  $c$ = {c_bin:.8f}',
        fontsize=10, color='#1A1A2E', pad=6)
    plt.tight_layout()
    fname = f'sensitivity_lam{tag}_binarized.pdf'
    fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved: {fname}')

print('\n✓ 所有图片已单独保存')


# ══════════════════════════════════════════════════════════════════════════════
#  数值汇总
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print(f"  {'lambda_pf':<10}  {'c (continuous)':>22}  {'c (binarized)':>22}  {'binary ratio':>14}")
print("-"*80)
for rec in records:
    print(f"  {rec['lam']:<10.3f}  {rec['c_cont']:>22.8f}  {rec['c_bin']:>22.8f}  {rec['bin_ratio']:>13.4%}")
print("="*80)