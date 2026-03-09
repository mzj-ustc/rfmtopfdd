"""
Ablation study: fixed total parameters ≈ 9600
Configs: 1×1, 2×2, 3×3, 4×4  (n_depths=1 fixed)
Conditions: with / without reg_term
Outputs:
  - results.csv  (compliance_raw, compliance_binary, 10 decimal places)
  - structure images (one per config×condition×type)
"""

import os, csv, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import label
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ─── device ───────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

os.makedirs('results', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Problem
# ══════════════════════════════════════════════════════════════════════════════
class Problems:
    def dlX_disp(self):
        n = self.batch_size - self.dlX_fixed.shape[0] - self.dlX_force.shape[0]
        domain_ycoord = np.random.uniform(-self.nely/(2*self.nelm),  self.nely/(2*self.nelm), (n,1))
        domain_xcoord = np.random.uniform(-self.nelx/(2*self.nelm),  self.nelx/(2*self.nelm), (n,1))
        domain_zcoord = np.random.uniform(-self.nelz/(2*self.nelm),  self.nelz/(2*self.nelm), (n,1))
        domain_coord  = np.concatenate((domain_ycoord, domain_xcoord, domain_zcoord), axis=1)
        coord = np.concatenate((self.dlX_fixed.cpu().detach().numpy(),
                                self.dlX_force.cpu().detach().numpy()), axis=0)
        coord = np.concatenate((coord, domain_coord), axis=0)
        return torch.tensor(coord, dtype=torch.float32, requires_grad=True).to(device)


class Cantilever_Beam_3D(Problems):
    def __init__(self, nelx, nely, nelz, xid, yid, zid, vf):
        self.xid=xid; self.yid=yid; self.zid=zid
        self.nelx=nelx; self.nely=nely; self.nelz=nelz
        self.nele=nelx*nely*nelz; self.nelm=max(nelx,nely,nelz)
        self.volfrac=vf; self.E0=1000; self.nu=0.3
        self.batch_size=30000
        self.alpha_init=1; self.alpha_max=100; self.alpha_delta=0.5; self.penal=3.0

        c_y,c_x,c_z = np.meshgrid(
            np.linspace(-nely/(2*self.nelm), nely/(2*self.nelm), nely),
            np.linspace(-nelx/(2*self.nelm), nelx/(2*self.nelm), nelx),
            np.linspace(-nelz/(2*self.nelm), nelz/(2*self.nelm), nelz), indexing='ij')
        self.dlX = np.stack((c_y.reshape(-1), c_x.reshape(-1), c_z.reshape(-1)), axis=1)

        c_y,c_x,c_z = np.meshgrid(
            np.linspace(-nely/(2*self.nelm), nely/(2*self.nelm), 2*nely),
            np.linspace(-nelx/(2*self.nelm), nelx/(2*self.nelm), 2*nelx),
            np.linspace(-nelz/(2*self.nelm), nelz/(2*self.nelm), 2*nelz), indexing='ij')
        self.dlXSS = np.stack((c_y.reshape(-1), c_x.reshape(-1), c_z.reshape(-1)), axis=1)

        self.V = ((np.max(self.dlX[:,0])-np.min(self.dlX[:,0]))*
                  (np.max(self.dlX[:,1])-np.min(self.dlX[:,1]))*
                  (np.max(self.dlX[:,2])-np.min(self.dlX[:,2])))

        fixed_voxel = np.zeros((nely,nelx,nelz)); fixed_voxel[:,0,:]=1.0
        dlX_fixed = self.dlX[np.where(fixed_voxel.reshape(-1,1)==1.0)[0],:]

        self.F_vector = torch.tensor([[0.1],[0.0],[0.0]], dtype=torch.float32).to(device)
        self.force_voxel = np.zeros((nely,nelx,nelz)); self.force_voxel[yid,xid,zid]=1
        dlX_force = self.dlX[np.where(self.force_voxel.reshape(-1,1)==1)[0],:]

        self.dlX      = torch.tensor(self.dlX,    dtype=torch.float32, requires_grad=True).to(device)
        self.dlXSS    = torch.tensor(self.dlXSS,  dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_fixed= torch.tensor(dlX_fixed,   dtype=torch.float32, requires_grad=True).to(device)
        self.dlX_force= torch.tensor(dlX_force,   dtype=torch.float32, requires_grad=True).to(device)

    def analytical_fixed_BC(self, u, coord):
        return u * 2*(1/(1+torch.exp(-20*(coord[:,1:2]+0.5)))-0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Spectral components
# ══════════════════════════════════════════════════════════════════════════════
_BAND_LOW, _BAND_HIGH = 0.0, 35.0

def _build_spectral_kernel(n_basis, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=None):
    rng = np.random.default_rng(seed)
    def _s1d(n):
        u = rng.uniform(0,1,n); half=band_high-band_low; val=u*2*half
        return np.where(val<half, val-band_high, val-half+band_low)
    ky,kx,kz = _s1d(n_basis),_s1d(n_basis),_s1d(n_basis)
    return torch.tensor(np.stack((ky,kx,kz),axis=0), dtype=torch.float32)


class _SpectralBase(nn.Module):
    def __init__(self, trainable_kernel, n_basis=100, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=None):
        super().__init__()
        K = _build_spectral_kernel(n_basis, band_low, band_high, seed).to(device)
        if trainable_kernel: self.kernel1 = nn.Parameter(K)
        else: self.register_buffer('kernel1', K)
        self.register_buffer('_bias', torch.ones(1, K.shape[1], dtype=torch.float32).to(device))
    def _features(self, x): return torch.sin(torch.matmul(x, self.kernel1)+self._bias)


class Disp_Net(_SpectralBase):
    def __init__(self, n_basis=500, band_low=_BAND_LOW, band_high=_BAND_HIGH, seed=42):
        super().__init__(False, n_basis, band_low, band_high, seed)
    def forward(self, x): return self._features(x)


class TO_BlockNet(nn.Module):
    def __init__(self, n_cols=2, n_rows=2, n_depths=1, n_basis=50,
                 band_low=_BAND_LOW, band_high=_BAND_HIGH, overlap=0.0, seed=None):
        super().__init__()
        self.n_cols=int(n_cols); self.n_rows=int(n_rows); self.n_depths=int(n_depths)
        self.n_blocks=self.n_cols*self.n_rows*self.n_depths
        self.overlap=overlap
        self.bias = nn.Parameter(torch.full((self.n_blocks,), math.log(0.3/0.7), device=device))
        K = _build_spectral_kernel(n_basis, band_low, band_high, seed).to(device)
        self.M = K.shape[1]
        self.kernels = nn.Parameter(K.unsqueeze(0).expand(self.n_blocks,-1,-1).clone())
        self.weights = nn.Parameter(torch.zeros(self.n_blocks, self.M, device=device))

    def _block_indices(self, coords):
        ys,xs,zs = coords[:,0],coords[:,1],coords[:,2]
        my,My=ys.min(),ys.max(); mx,Mx=xs.min(),xs.max(); mz,Mz=zs.min(),zs.max()
        row=((ys-my)/(My-my+1e-9)*self.n_rows).long().clamp(0,self.n_rows-1)
        col=((xs-mx)/(Mx-mx+1e-9)*self.n_cols).long().clamp(0,self.n_cols-1)
        dep=((zs-mz)/(Mz-mz+1e-9)*self.n_depths).long().clamp(0,self.n_depths-1)
        return row*(self.n_cols*self.n_depths)+col*self.n_depths+dep

    def _soft_weights(self, coords):
        ys=coords[:,0:1]; xs=coords[:,1:2]; zs=coords[:,2:3]
        my,My=ys.min().item(),ys.max().item(); mx,Mx=xs.min().item(),xs.max().item()
        mz,Mz=zs.min().item(),zs.max().item()
        bh=(My-my)/self.n_rows; bw=(Mx-mx)/self.n_cols; bd=(Mz-mz)/self.n_depths
        cy=torch.tensor([my+(r+.5)*bh for r in range(self.n_rows)
                         for c in range(self.n_cols) for d in range(self.n_depths)],
                        dtype=torch.float32, device=device)
        cx=torch.tensor([mx+(c+.5)*bw for r in range(self.n_rows)
                         for c in range(self.n_cols) for d in range(self.n_depths)],
                        dtype=torch.float32, device=device)
        cz=torch.tensor([mz+(d+.5)*bd for r in range(self.n_rows)
                         for c in range(self.n_cols) for d in range(self.n_depths)],
                        dtype=torch.float32, device=device)
        sy=bh*(0.5+self.overlap); sx=bw*(0.5+self.overlap); sz=bd*(0.5+self.overlap)
        dy=(ys-cy.unsqueeze(0))/sy; dx=(xs-cx.unsqueeze(0))/sx; dz=(zs-cz.unsqueeze(0))/sz
        return torch.softmax(-0.5*(dy**2+dx**2+dz**2), dim=1)

    def _all_blocks(self, coords):
        N,B,M = coords.shape[0],self.n_blocks,self.M
        K_flat=self.kernels.permute(1,0,2).reshape(3,B*M)
        z=(coords@K_flat).view(N,B,M)+1.0
        out=(torch.sin(z)*self.weights.unsqueeze(0)).sum(dim=2)+self.bias.unsqueeze(0)
        return torch.sigmoid(out)

    def forward(self, coords):
        out_all = self._all_blocks(coords)
        if self.overlap<=0:
            bidx=self._block_indices(coords)
            N=coords.shape[0]
            return out_all[torch.arange(N,device=device), bidx].unsqueeze(-1)
        else:
            w=self._soft_weights(coords)
            return (out_all*w).sum(dim=1, keepdim=True)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# Analytical basis matrix
# ══════════════════════════════════════════════════════════════════════════════
def cal_matrix_3d(model, points, dlx_force, M):
    K = model.kernel1
    with torch.no_grad():
        z=torch.matmul(points,K)+model._bias
        sin_z=torch.sin(z); cos_z=torch.cos(z)
        x1=points[:,1:2]; sigma=torch.sigmoid(20.0*(x1+0.5)); bc=2.0*(sigma-0.5)
        d_bc=40.0*sigma*(1.0-sigma)
        u1  = sin_z*bc
        u_y = cos_z*K[0:1,:]*bc
        u_x = cos_z*K[1:2,:]*bc + sin_z*d_bc
        u_z = cos_z*K[2:3,:]*bc
        v_force = model(dlx_force)
    return (u1.cpu().numpy(), u_x.cpu().numpy(), u_y.cpu().numpy(),
            u_z.cpu().numpy(), v_force.cpu().numpy())


# ══════════════════════════════════════════════════════════════════════════════
# PDE loss
# ══════════════════════════════════════════════════════════════════════════════
_E_MOD,_NU_MOD = 1000.0,0.3
_LAME_MU       = _E_MOD/(2.0*(1.0+_NU_MOD))
_LAME_LAMBDA   = _E_MOD*_NU_MOD/((1.0+_NU_MOD)*(1.0-2.0*_NU_MOD))

def pinnloss3d(weights1, u_x, u_y, u_z, v_force, problem, xPhys_m):
    xPhys_m=xPhys_m.reshape(-1,1)
    dux_dx=(u_x@weights1[:,1]).reshape(-1,1); dux_dy=(u_y@weights1[:,1]).reshape(-1,1)
    dux_dz=(u_z@weights1[:,1]).reshape(-1,1)
    duy_dx=(u_x@weights1[:,0]).reshape(-1,1); duy_dy=(u_y@weights1[:,0]).reshape(-1,1)
    duy_dz=(u_z@weights1[:,0]).reshape(-1,1)
    duz_dx=(u_x@weights1[:,2]).reshape(-1,1); duz_dy=(u_y@weights1[:,2]).reshape(-1,1)
    duz_dz=(u_z@weights1[:,2]).reshape(-1,1)
    eps11=dux_dx; eps22=duy_dy; eps33=duz_dz
    eps12=0.5*(dux_dy+duy_dx); eps13=0.5*(dux_dz+duz_dx); eps23=0.5*(duy_dz+duz_dy)
    trace_sq=(eps11+eps22+eps33)**2
    diag_sq =eps11**2+eps22**2+eps33**2
    energy=(0.5*_LAME_LAMBDA*trace_sq+_LAME_MU*(diag_sq+2*eps12**2+2*eps13**2+2*eps23**2))
    energy=energy*(xPhys_m**3.0)
    energy_ans=problem.V*torch.mean(energy)
    force_l=torch.mean(0.1*(v_force@weights1[:,0]).reshape(-1,1))
    return energy_ans-force_l, energy


# ══════════════════════════════════════════════════════════════════════════════
# Interface continuity loss
# ══════════════════════════════════════════════════════════════════════════════
def continueloss(to_model, problem):
    n_samples=500; coords_all=problem.dlX
    min_y,max_y=torch.min(coords_all[:,0]).item(),torch.max(coords_all[:,0]).item()
    min_x,max_x=torch.min(coords_all[:,1]).item(),torch.max(coords_all[:,1]).item()
    min_z,max_z=torch.min(coords_all[:,2]).item(),torch.max(coords_all[:,2]).item()
    n_side=max(2,int(n_samples**0.5))
    losses=[]; nc,nr,nd=to_model.n_cols,to_model.n_rows,to_model.n_depths
    def _b(r,c,d): return r*(nc*nd)+c*nd+d

    for k in range(nc-1):
        ix=min_x+(k+1)*(max_x-min_x)/nc
        ys=torch.linspace(min_y,max_y,n_side,device=device)
        zs=torch.linspace(min_z,max_z,n_side,device=device)
        yf,zf=(t.flatten().unsqueeze(1) for t in torch.meshgrid(ys,zs,indexing='ij'))
        pts=torch.cat([yf,torch.ones_like(yf)*ix,zf],dim=1)
        ao=to_model._all_blocks(pts)
        ri=((yf.squeeze()-min_y)/(max_y-min_y+1e-9)*nr).long().clamp(0,nr-1)
        di=((zf.squeeze()-min_z)/(max_z-min_z+1e-9)*nd).long().clamp(0,nd-1)
        for r in range(nr):
            for d in range(nd):
                m=(ri==r)&(di==d)
                if m.sum()==0: continue
                losses.append(torch.mean((ao[m,_b(r,k,d)]-ao[m,_b(r,k+1,d)])**2))

    for k in range(nr-1):
        iy=min_y+(k+1)*(max_y-min_y)/nr
        xs=torch.linspace(min_x,max_x,n_side,device=device)
        zs=torch.linspace(min_z,max_z,n_side,device=device)
        xf,zf=(t.flatten().unsqueeze(1) for t in torch.meshgrid(xs,zs,indexing='ij'))
        pts=torch.cat([torch.ones_like(xf)*iy,xf,zf],dim=1)
        ao=to_model._all_blocks(pts)
        ci=((xf.squeeze()-min_x)/(max_x-min_x+1e-9)*nc).long().clamp(0,nc-1)
        di=((zf.squeeze()-min_z)/(max_z-min_z+1e-9)*nd).long().clamp(0,nd-1)
        for c in range(nc):
            for d in range(nd):
                m=(ci==c)&(di==d)
                if m.sum()==0: continue
                losses.append(torch.mean((ao[m,_b(k,c,d)]-ao[m,_b(k+1,c,d)])**2))

    for k in range(nd-1):
        iz=min_z+(k+1)*(max_z-min_z)/nd
        ys=torch.linspace(min_y,max_y,n_side,device=device)
        xs=torch.linspace(min_x,max_x,n_side,device=device)
        yf,xf=(t.flatten().unsqueeze(1) for t in torch.meshgrid(ys,xs,indexing='ij'))
        pts=torch.cat([yf,xf,torch.ones_like(yf)*iz],dim=1)
        ao=to_model._all_blocks(pts)
        ri=((yf.squeeze()-min_y)/(max_y-min_y+1e-9)*nr).long().clamp(0,nr-1)
        ci=((xf.squeeze()-min_x)/(max_x-min_x+1e-9)*nc).long().clamp(0,nc-1)
        for r in range(nr):
            for c in range(nc):
                m=(ri==r)&(ci==c)
                if m.sum()==0: continue
                losses.append(torch.mean((ao[m,_b(r,c,k)]-ao[m,_b(r,c,k+1)])**2))

    if not losses: return torch.tensor(0.0,device=device)
    return torch.stack(losses).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Connectivity filter
# ══════════════════════════════════════════════════════════════════════════════
def filter_disconnected_regions(density_matrix, threshold=0.5):
    binary_matrix=( density_matrix>threshold).astype(int)
    labeled_matrix,nf=label(binary_matrix)
    lbl_max,n_max=0,0
    for lbl in range(1,nf+1):
        n=np.sum(labeled_matrix==lbl)
        if n>n_max: lbl_max,n_max=lbl,n
    filtered=np.zeros_like(density_matrix)
    filtered[labeled_matrix==lbl_max]=density_matrix[labeled_matrix==lbl_max]
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════
class RFM_TONN:
    def __init__(self, problem, to_model, disp_model, use_reg=True):
        self.problem=problem; self.disp_model=disp_model
        self.to_model=to_model; self.use_reg=use_reg
        self.total_epoch=0; self.log_pinn_init_loss=[]

        self.to_optimizer=optim.Adam(self.to_model.parameters(), lr=0.03)
        self.to_scheduler=optim.lr_scheduler.LinearLR(
            self.to_optimizer, start_factor=1.0, end_factor=1e-2/3e-2, total_iters=400)

        self.coord=problem.dlX_disp()
        M=disp_model.kernel1.shape[1]
        _,u_x,u_y,u_z,v_force=cal_matrix_3d(self.disp_model,self.coord,self.problem.dlX_force,M)
        self.u_x    =torch.tensor(u_x,    dtype=torch.float32,device=device)
        self.u_y    =torch.tensor(u_y,    dtype=torch.float32,device=device)
        self.u_z    =torch.tensor(u_z,    dtype=torch.float32,device=device)
        self.v_force=torch.tensor(v_force,dtype=torch.float32,device=device)

    def fit_disp_init(self):
        M=self.disp_model.kernel1.shape[1]
        self.weights1=nn.Parameter(torch.zeros(M,3,dtype=torch.float32,device=device))
        self.disp_optimizer=optim.Adam([self.weights1],lr=5e-6)
        xPhys_init=torch.ones(self.coord.shape[0],1,device=device)*0.3
        for _ in range(1000):
            loss,_=pinnloss3d(self.weights1,self.u_x,self.u_y,self.u_z,
                               self.v_force,self.problem,xPhys_init)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

        _,u_x,u_y,u_z,v_force=cal_matrix_3d(
            self.disp_model,self.problem.dlX,self.problem.dlX_force,M)
        u_x=torch.tensor(u_x,dtype=torch.float32,device=device)
        u_y=torch.tensor(u_y,dtype=torch.float32,device=device)
        u_z=torch.tensor(u_z,dtype=torch.float32,device=device)
        v_force=torch.tensor(v_force,dtype=torch.float32,device=device)
        xPhys=torch.ones(self.problem.dlX.shape[0],1,device=device)*0.3
        loss,energy_c=pinnloss3d(self.weights1,u_x,u_y,u_z,v_force,self.problem,xPhys)
        self.c_0=torch.mean(energy_c)

    def to_loss(self, coord):
        self.total_epoch+=1
        xPhys_m=self.to_model(coord)
        alpha=min(self.problem.alpha_init+self.problem.alpha_delta*self.total_epoch,
                  self.problem.alpha_max)
        _,energy_c=pinnloss3d(self.weights1,self.u_x,self.u_y,self.u_z,
                               self.v_force,self.problem,xPhys_m)

        class ComputeDeDrho(torch.autograd.Function):
            @staticmethod
            def forward(ctx,xPhys_m,energy_c,coord):
                ctx.save_for_backward(xPhys_m,energy_c,coord); return energy_c
            @staticmethod
            def backward(ctx,denergy):
                xPhys_m,energy_c,coord=ctx.saved_tensors
                grad=torch.autograd.grad(energy_c,xPhys_m,grad_outputs=denergy,
                                         create_graph=True,retain_graph=True)[0]
                return (-grad,torch.zeros_like(energy_c),torch.zeros_like(coord))

        c=torch.mean(ComputeDeDrho.apply(xPhys_m,energy_c,coord))
        xPhys_dlX=self.to_model(self.problem.dlX)
        vf=torch.mean(xPhys_dlX)

        loss=alpha*(vf/self.problem.volfrac-1.0)**2 + c/self.c_0.detach()

        if self.use_reg:
            epsilon=8e-3; gamma=0.1
            grad_phi=torch.autograd.grad(
                xPhys_m.sum(),coord,create_graph=True,retain_graph=True)[0]
            grad_term=0.5*epsilon*torch.mean(torch.sum(grad_phi**2,dim=1))
            double_well=(1/epsilon)*torch.mean(xPhys_m**2*(1-xPhys_m)**2)
            reg_term=gamma*(0.01*grad_term+double_well)
            loss=loss+0.1*reg_term

        if self.total_epoch % 50 == 0:
            print(f'  Epoch {self.total_epoch:4d}  loss={loss.item():.5f}  '
                  f'c={c.item():.5f}  vf={vf.item():.4f}')
        return loss

    def fit_disp(self, epochs=200):
        for _ in range(epochs):
            xPhys_m=self.to_model(self.coord)
            loss,_=pinnloss3d(self.weights1,self.u_x,self.u_y,self.u_z,
                               self.v_force,self.problem,xPhys_m)
            self.disp_optimizer.zero_grad(); loss.backward(); self.disp_optimizer.step()

    def fit_to(self, epochs=400):
        for epoch in range(epochs):
            self.fit_disp(50)
            loss=self.to_loss(self.coord)
            self.to_optimizer.zero_grad(); loss.backward()
            self.to_optimizer.step(); self.to_scheduler.step()


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation (no title/axis/colorbar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_iso_clean(xPhys, filepath):
    """Marching-cubes render, clean (no title, axes, colorbar)."""
    nely,nelx,nelz=xPhys.shape; nelm=max(nely,nelx,nelz)
    padding=np.zeros([nely+2,nelx+2,nelz+2])
    padding[1:-1,1:-1,1:-1]=xPhys
    try:
        verts,faces,normals,values=measure.marching_cubes(padding, 0.5)
    except Exception:
        fig=plt.figure(figsize=(8,5)); plt.axis('off')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(); return

    fig=plt.figure(figsize=(10,6))
    ax=fig.add_subplot(1,1,1,projection='3d')
    ls=LightSource(azdeg=45,altdeg=-45)
    f_coord=np.take(verts,faces,axis=0)
    f_norm=np.cross(f_coord[:,2]-f_coord[:,0],f_coord[:,1]-f_coord[:,0])
    cl=ls.shade_normals(f_norm)
    norm=colors.Normalize(vmin=0.0,vmax=1.0,clip=True)
    mapper=cm.ScalarMappable(norm=norm,cmap=cm.gray_r)
    rgb=mapper.to_rgba(cl).reshape([-1,4])
    mesh=Poly3DCollection(np.take(verts,faces,axis=0)/nelx*
                          np.array([[nelm/nely,nelm/nelm,nelm/nelz]]))
    ax.add_collection3d(mesh); mesh.set_facecolors(rgb)
    ax.view_init(-150,-120,vertical_axis='x')
    ax.set_box_aspect(aspect=(nelx,nelz,nely))
    ax.set_axis_off()
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEM compliance
# ══════════════════════════════════════════════════════════════════════════════
def lk_H8(nu):
    A=np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],
                [-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]])
    k=(1/144)*A.T@np.array([1,nu])
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
    KE=np.block([[K1,K2,K3,K4],[K2.T,K5,K6,K3.T],[K3.T,K6.T,K5.T,K2.T],[K4,K3,K2,K1.T]])
    return KE/((nu+1)*(1-2*nu))

def compute_compliance(xPhys, nelx, nely, nelz, E0=1000.0, Emin=1e-9, nu=0.3):
    nele=nelx*nely*nelz; ndof=3*(nelx+1)*(nely+1)*(nelz+1); KE=lk_H8(nu)
    j_idx,k_idx=np.meshgrid(np.arange(nely+1),np.arange(nelz+1),indexing='ij')
    fixednid=(k_idx*(nely+1)*(nelx+1)+j_idx).flatten()
    fixeddof=np.concatenate([3*fixednid,3*fixednid+1,3*fixednid+2])
    freedofs=np.setdiff1d(np.arange(ndof),fixeddof)
    loadnid=(nelz//2)*(nelx+1)*(nely+1)+nelx*(nely+1)+nely//2
    loaddof=3*loadnid+1
    F=sparse.lil_matrix((ndof,1)); F[loaddof,0]=-0.1; F=F.tocsr()
    nodegrd=np.reshape(np.arange(1,(nely+1)*(nelx+1)+1),(nely+1,nelx+1),order='F')
    nodeids=np.reshape(nodegrd[:-1,:-1],(nely*nelx,1),order='F')
    nodeidz=np.arange(0,nelz*(nely+1)*(nelx+1),(nely+1)*(nelx+1))
    nodeids=(np.tile(nodeids,(1,nelz))+np.tile(nodeidz,(nely*nelx,1))).flatten(order='F')
    edofVec=3*(nodeids-1); n_layer=(nely+1)*(nelx+1)
    offset_node=np.array([0,nely+1,nely+2,1,n_layer,n_layer+nely+1,n_layer+nely+2,n_layer+1])
    edofMat=np.zeros((nele,24),dtype=int)
    for i in range(24): edofMat[:,i]=edofVec+3*offset_node[i//3]+(i%3)
    iK=np.repeat(edofMat,24,axis=1).flatten(); jK=np.repeat(edofMat,24,axis=0).flatten()
    xf=xPhys.flatten(order='F')
    sK=(KE.ravel()[np.newaxis,:]*(Emin+xf**3*(E0-Emin))[:,np.newaxis]).ravel()
    K=sparse.coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsr(); K=(K+K.T)*0.5
    U=np.zeros(ndof)
    U[freedofs]=spsolve(K[np.ix_(freedofs,freedofs)],F[freedofs].toarray().ravel())
    Ue=U[edofMat]; ce=np.sum((Ue@KE)*Ue,axis=1)
    return np.sum((Emin+xf**3*(E0-Emin))*ce)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment configurations
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_PARAMS_TARGET = 9600

# blocks: (n_cols, n_rows, n_depths=1)
block_configs = [(2,2,2), (3,3,3), (4,4,4)]

# For each config, compute M such that total params ≈ TOTAL_PARAMS_TARGET
# TO_BlockNet params: B*(3*M + M + 1) = B*(4M+1)
# M = (TOTAL_PARAMS_TARGET/B - 1) // 4
def calc_n_basis(n_cols, n_rows, n_depths, target=TOTAL_PARAMS_TARGET):
    B = n_cols * n_rows * n_depths
    M = max(1, int((target / B - 1) // 4))
    return M

# Problem setup
nelx,nely,nelz = 60,20,8
xid,yid,zid,vf = 59,9,3,0.3

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
csv_rows = []

for (nc, nr, nd) in block_configs:
    n_basis_to = calc_n_basis(nc, nr, nd)
    B = nc*nr*nd
    actual_params = B*(4*n_basis_to+1)
    config_str = f'{nc}x{nr}'
    print(f'\n{"="*60}')
    print(f'Block config: {config_str}  (B={B}, M={n_basis_to}, params≈{actual_params})')

    for use_reg in [True, False]:
        reg_str = 'reg' if use_reg else 'noreg'
        print(f'\n--- {config_str} | {reg_str} ---')

        problem = Cantilever_Beam_3D(nelx, nely, nelz, xid, yid, zid, vf)

        to_model = TO_BlockNet(
            n_cols=nc, n_rows=nr, n_depths=nd,
            n_basis=n_basis_to, band_low=0.0, band_high=35.0, overlap=0.2
        ).to(device)

        disp_model = Disp_Net(n_basis=8000, band_low=0.0, band_high=35.0, seed=42).to(device)

        solver = RFM_TONN(problem, to_model, disp_model, use_reg=use_reg)

        t0 = time.time()
        solver.fit_disp_init()
        solver.fit_to(epochs=400)
        elapsed = time.time()-t0
        print(f'  Training done in {elapsed:.1f}s')

        # ── Extract density field ──
        with torch.no_grad():
            xPhys_dlX = solver.to_model(problem.dlX)
        tt = xPhys_dlX.cpu().detach().numpy().reshape(nely, nelx, nelz)
        tt = filter_disconnected_regions(tt, threshold=0.5)
        tt_binary = (tt > 0.4).astype(float)

        # ── Compliance ──
        c_raw    = compute_compliance(tt,        nelx, nely, nelz)
        c_binary = compute_compliance(tt_binary, nelx, nely, nelz)
        print(f'  compliance_raw={c_raw:.10f}  compliance_binary={c_binary:.10f}')

        csv_rows.append({
            'config': config_str,
            'n_blocks': B,
            'n_basis': n_basis_to,
            'actual_params': actual_params,
            'use_reg': use_reg,
            'compliance_raw': f'{c_raw:.10f}',
            'compliance_binary': f'{c_binary:.10f}',
            'train_time_s': f'{elapsed:.1f}',
        })

        # ── Super-sampled density for visualisation ──
        with torch.no_grad():
            xPhys_ss = solver.to_model(problem.dlXSS)
        tt_ss = xPhys_ss.cpu().detach().numpy().reshape(2*nely, 2*nelx, 2*nelz)
        tt_ss = filter_disconnected_regions(tt_ss, threshold=0.5)
        tt_ss_bin = (tt_ss > 0.4).astype(float)

        # ── Save structure plots ──
        fname_raw    = f'results/struct_{config_str}_{reg_str}_raw.png'
        fname_binary = f'results/struct_{config_str}_{reg_str}_binary.png'
        plot_iso_clean(tt_ss,     fname_raw)
        plot_iso_clean(tt_ss_bin, fname_binary)
        print(f'  Saved: {fname_raw}')
        print(f'  Saved: {fname_binary}')

        # ── Free GPU memory ──
        del solver, to_model, disp_model, problem
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ══════════════════════════════════════════════════════════════════════════════
# Save CSV
# ══════════════════════════════════════════════════════════════════════════════
csv_path = 'results/compliance_results.csv'
fieldnames = ['config','n_blocks','n_basis','actual_params','use_reg',
              'compliance_raw','compliance_binary','train_time_s']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f'\n✓ Results saved to {csv_path}')
print(f'✓ Structure images saved in results/')

# ── Print summary table ──
print('\n' + '='*80)
print(f'{"Config":<8} {"UseReg":<8} {"Params":<8} {"Compliance(raw)":<22} {"Compliance(binary)":<22}')
print('-'*80)
for row in csv_rows:
    print(f'{row["config"]:<8} {str(row["use_reg"]):<8} {row["actual_params"]:<8} '
          f'{row["compliance_raw"]:<22} {row["compliance_binary"]:<22}')
