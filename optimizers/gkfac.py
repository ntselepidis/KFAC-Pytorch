import math

import torch

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat
from utils.kfac_utils import get_matrix_form_grad


class GKFAC(torch.optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 solver='symeig',
                 omega_1=1.0,
                 omega_2=1.0,
                 mode='nearest',
                 device='cuda'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)

        # TODO (CW): GKFAC optimizer now only support model as input
        super(GKFAC, self).__init__(model.parameters(), defaults)

        self.model = model
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self._register_modules()

        # utility vars
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.stat_decay = 0 # stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.steps = 0

        # one-level KFAC vars
        self.solver = solver
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.Inv_a, self.Inv_g = {}, {}

        # two-level KFAC vars
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.mode = mode
        self.nlayers = len(self.modules)
        self.a = [[] for l in range(self.nlayers)]
        self.g = [[] for l in range(self.nlayers)]
        self.sum_aa = torch.zeros(self.nlayers, self.nlayers, device=device)
        self.sum_gg = torch.zeros(self.nlayers, self.nlayers, device=device)

    @staticmethod
    def _downsample_multiply(a, i, j, batch_size, mode='nearest'):
        # Get spatial dimensions of a[i] and a[j]
        spatial_dim_i = int(math.sqrt( a[i].shape[0] / batch_size ))
        spatial_dim_j = int(math.sqrt( a[j].shape[0] / batch_size ))
        if (spatial_dim_i == spatial_dim_j):
            cov_ij = a[i].t() @ (a[j] / batch_size)
        elif (spatial_dim_j > spatial_dim_i):
            a_j_dsmpl = a[j].view(batch_size, spatial_dim_j, spatial_dim_j, -1).permute(0, 3, 1, 2)
            a_j_dsmpl = torch.nn.functional.interpolate(a_j_dsmpl, (spatial_dim_i, spatial_dim_i), mode=mode).permute(0, 2, 3, 1)
            a_j_dsmpl = a_j_dsmpl.reshape(-1, a_j_dsmpl.size(-1))
            cov_ij = a[i].t() @ (a_j_dsmpl / batch_size)
        else:
            a_i_dsmpl = a[i].view(batch_size, spatial_dim_i, spatial_dim_i, -1).permute(0, 3, 1, 2)
            a_i_dsmpl = torch.nn.functional.interpolate(a_i_dsmpl, (spatial_dim_j, spatial_dim_j), mode=mode).permute(0, 2, 3, 1)
            a_i_dsmpl = a_i_dsmpl.reshape(-1, a_i_dsmpl.size(-1))
            cov_ij = a_i_dsmpl.t() @ (a[j] / batch_size)
        return cov_ij

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            # Get module index
            i = self.modules.index(module)
            with torch.no_grad():
                aa, self.a[i] = self.CovAHandler(input[0], module)
            # Initialize buffer
            if self.steps == 0:
                self.m_aa[module] = torch.zeros_like(aa)
            update_running_stat(aa, self.m_aa[module], self.stat_decay)
            self.sum_aa[i][i] = self.m_aa[module].sum().item()
            # Update sums of off-diagonal blocks of A
            for j in range(i):
                # Compute inter-layer covariances (downsample if needed)
                new_aa = self._downsample_multiply(self.a, i, j, input[0].shape[0], self.mode)
                # Update sum
                self.sum_aa[i][j] *= self.stat_decay
                self.sum_aa[i][j] += (1 - self.stat_decay) * new_aa.sum().item()

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            # Get module index
            i = self.modules.index(module)
            gg, self.g[i] = self.CovGHandler(grad_output[0], module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.zeros_like(gg)
            update_running_stat(gg, self.m_gg[module], self.stat_decay)
            self.sum_gg[i][i] = self.m_gg[module].sum().item()
            # Update sums of off-diagonal blocks of G
            for j in range(i, self.nlayers):
                # Compute inter-layer covariances (downsample if needed)
                new_gg = self._downsample_multiply(self.g, i, j, grad_output[0].shape[0], self.mode)
                # Update sum
                self.sum_gg[i][j] *= self.stat_decay
                self.sum_gg[i][j] += (1 - self.stat_decay) * new_gg.sum().item()

    def _register_modules(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in GKFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in GKFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition or approximate factorization for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        if self.solver == 'symeig':
            eps = 1e-10  # for numerical stability
            self.d_a[m], self.Q_a[m] = torch.symeig(
                self.m_aa[m], eigenvectors=True)
            self.d_g[m], self.Q_g[m] = torch.symeig(
                self.m_gg[m], eigenvectors=True)

            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())
        else:
            group = self.param_groups[0]
            damping = group['damping']
            numer = self.m_aa[m].trace().item() * self.m_gg[m].shape[0]
            denom = self.m_gg[m].trace().item() * self.m_aa[m].shape[0]
            pi = numer / denom
            assert numer > 0, "trace(A) should be positive"
            assert denom > 0, "trace(G) should be positive"
            # assert pi > 0, "pi should be positive"
            diag_a = self.m_aa[m].new(self.m_aa[m].shape[0]).fill_((damping * pi)**0.5)
            diag_g = self.m_gg[m].new(self.m_gg[m].shape[0]).fill_((damping / pi)**0.5)
            self.Inv_a[m] = ( self.m_aa[m] + torch.diag(diag_a) ).inverse()
            self.Inv_g[m] = ( self.m_gg[m] + torch.diag(diag_g) ).inverse()

    def _update_coarse_fisher_inv(self):
        group = self.param_groups[0]
        damping = group['damping']
        self.sum_aa += self.sum_aa.t() - self.sum_aa.diag().diag()
        self.sum_gg += self.sum_gg.t() - self.sum_gg.diag().diag()
        coarse_damping = self.sum_aa.new(self.sum_aa.shape[0]).fill_(damping)
        self.coarse_F_inverse = ( self.sum_aa * self.sum_gg + torch.diag(coarse_damping)).inverse()

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        if self.solver == 'symeig':
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        else:
            v = self.Inv_g[m] @ p_grad_mat @ self.Inv_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.size())
            v[1] = v[1].view(m.bias.grad.size())
        else:
            v = [v.view(m.weight.grad.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad * lr ** 2).sum().item()
        assert vg_sum != 0, "vg_sum should be non-zero"
        assert vg_sum > 0, "vg_sum should be positive"
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))
        # update grad
        for m in self.modules:
            v = updates[m]
            m.weight.grad.copy_(v[0])
            m.weight.grad.mul_(nu)
            if m.bias is not None:
                m.bias.grad.copy_(v[1])
                m.bias.grad.mul_(nu)

    @torch.no_grad()
    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0: #and self.steps >= 20 * self.TCov:
                    d_p.add_(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-group['lr'])

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        # Update 2-level preconditioner
        if self.steps % self.TInv == 0:
            # Update layer inverses (diagonal blocks of fine Fisher)
            for m in self.modules:
                self._update_inv(m)
            # Recompute and invert coarse Fisher
            self._update_coarse_fisher_inv()
        # Compute fine part of natural gradient and assemble coarse rhs
        updates = {}
        coarse_rhs = torch.zeros(self.nlayers, 1, device=self.coarse_F_inverse.device)
        for m_index, m in enumerate(self.modules):
            p_grad_mat = get_matrix_form_grad(m)
            coarse_rhs[m_index] = p_grad_mat.sum().item()
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        # Compute coarse part of natural gradient
        coarse_v = self.coarse_F_inverse @ coarse_rhs
        # Add fine and coarse parts of natural gradient
        for m_index, m in enumerate(self.modules):
            coarse_v_m = coarse_v[m_index]
            updates[m][0] = self.omega_1 * updates[m][0] + self.omega_2 * coarse_v_m
            if m.bias is not None:
                updates[m][1] = self.omega_1 * updates[m][1] + self.omega_2 * coarse_v_m

        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
        self.stat_decay = min( 1.0 - 1.0 / (self.steps // self.TCov + 1), 0.95 )
