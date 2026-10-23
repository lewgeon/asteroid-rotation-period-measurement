"""Scoped, optional CUDA kernels. CPU remains the reference and default."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import numpy as np

_active=ContextVar('inversion_compute_device',default='cpu')


def resolve_device(config):
    requested=config.get('compute_device','cpu')
    if requested not in ('cpu','cuda:0'):raise ValueError('compute_device必须为cpu或cuda:0')
    if requested=='cpu':return dict(requested='cpu',actual='cpu',reason=None)
    try:
        import torch
        if torch.cuda.is_available():return dict(requested=requested,actual='cuda:0',reason=None)
    except ImportError:pass
    return dict(requested=requested,actual='cpu',reason='请求GPU，但PyTorch CUDA不可用；本次明确回退CPU')


@contextmanager
def execution(config):
    state=resolve_device(config);token=_active.set(state['actual'])
    try:yield state
    finally:_active.reset(token)


def configured(config_index=1):
    def decorate(function):
        @wraps(function)
        def call(*args,**kwargs):
            config=args[config_index] if len(args)>config_index else kwargs['config']
            with execution(config):return function(*args,**kwargs)
        return call
    return decorate


def is_cuda():return _active.get()=='cuda:0'


def period_grid_sse(times, values, weights, group_columns, periods, harmonics, phase_offset):
    """Float64 batched small-system proposals; unsafe boxes use CPU SVD.

    Inputs have already passed the reference search's filtering/normalization.
    Returns SSEs, indices requiring the unchanged CPU solver and diagnostics.
    Local refinement deliberately stays on the reference CPU objective.
    """
    import torch
    n = len(times)
    p = group_columns.shape[1]+2*harmonics
    budget = 64*1024*1024
    # Design, weighted design, residual and small-system workspace allowance.
    batch = min(512, budget//max(1, 8*n*(4*p+6)))
    info = dict(kernel='float64_batched_normal_equations_CPU_SVD_guard', batch_size=int(batch),
                working_array_budget_bytes=budget, cpu_guard_grid_count=0)
    output = np.full(len(periods), np.nan)
    if batch < 1 or p > 64:
        info['reason'] = '单频设计矩阵超过GPU分块预算或参数数大于64；使用CPU参考求解'
        return output, np.arange(len(periods)), info
    try:
        with torch.inference_mode():
            t = torch.as_tensor(times, device='cuda:0', dtype=torch.float64)
            y = torch.as_tensor(values, device='cuda:0', dtype=torch.float64)
            w = torch.as_tensor(weights, device='cuda:0', dtype=torch.float64)
            root = torch.sqrt(w/w.mean())
            base = torch.as_tensor(group_columns, device='cuda:0', dtype=torch.float64)
            off = None if phase_offset is None else torch.as_tensor(phase_offset, device='cuda:0', dtype=torch.float64)
            for start in range(0, len(periods), batch):
                local = periods[start:start+batch]
                omega_np = 2*np.pi/local
                # Large arguments magnify backend range-reduction differences.
                with np.errstate(over='ignore', invalid='ignore'):
                    phase_bound = harmonics*(abs(omega_np)*np.max(abs(times)) +
                        (0. if phase_offset is None else np.max(abs(phase_offset))))
                safe_phase = np.isfinite(omega_np) & np.isfinite(phase_bound) & (phase_bound <= 1e6)
                ids = np.flatnonzero(safe_phase)
                if not len(ids): continue
                omega = torch.as_tensor(omega_np[ids], device='cuda:0', dtype=torch.float64)
                design = torch.empty((len(ids), n, p), device='cuda:0', dtype=torch.float64)
                design[:, :, :base.shape[1]] = base
                phase = omega[:, None]*t + (0 if off is None else off)
                for h in range(1, harmonics+1):
                    angle = (h*omega[:, None])*t if off is None else h*phase
                    design[:, :, base.shape[1]+2*(h-1)] = torch.cos(angle)
                    design[:, :, base.shape[1]+2*(h-1)+1] = torch.sin(angle)
                weighted = design*root[None, :, None]
                gram = weighted.transpose(1, 2) @ weighted
                eigenvalues = torch.linalg.eigvalsh(gram)
                # cond(A) < 1e3 keeps normal-equation amplification bounded;
                # the unchanged SVD reference handles every other grid point.
                safe = torch.isfinite(eigenvalues).all(dim=1) & (eigenvalues[:, 0] > eigenvalues[:, -1]*1e-6)
                ids_ok = torch.nonzero(safe).flatten()
                if not len(ids_ok): continue
                rhs = weighted[ids_ok].transpose(1, 2) @ (y*root)[None, :, None]
                beta, info_solve = torch.linalg.solve_ex(gram[ids_ok], rhs)
                error = y[None, :]-(design[ids_ok] @ beta).squeeze(-1)
                sse = torch.sum(w[None, :]*error.square(), dim=1).cpu().numpy()
                sse[info_solve.cpu().numpy()!=0] = np.nan
                output[start+ids[ids_ok.cpu().numpy()]] = sse
    except torch.cuda.OutOfMemoryError as error:
        raise MemoryError('GPU周期搜索显存不足；请选择CPU，未静默降低精度') from error
    guard = np.flatnonzero(~np.isfinite(output))
    return output, guard, info


def _budget(rows,width,arrays=6):
    # Bound each tensor batch; this is not a bound on all process/device memory.
    return max(1,min(rows,(64*1024*1024)//max(1,width*16*arrays)))


def matched_filter(iq,reference):
    import torch
    from scipy.fft import next_fast_len
    length=next_fast_len(iq.shape[1]+len(reference)-1)
    output=np.empty((len(iq),iq.shape[1]+len(reference)-1),dtype=np.complex128)
    kernel=torch.as_tensor(np.conj(reference[::-1]).copy(),device='cuda:0')
    transformed=torch.fft.fft(kernel,n=length)
    chunk=_budget(len(iq),length)
    try:
        with torch.inference_mode():
            for start in range(0,len(iq),chunk):
                values=torch.as_tensor(np.array(iq[start:start+chunk],copy=True),device='cuda:0')
                result=torch.fft.ifft(torch.fft.fft(values,n=length,dim=-1)*transformed,dim=-1)
                output[start:start+chunk]=result[:,:output.shape[1]].cpu().numpy()
    except torch.cuda.OutOfMemoryError as error:
        raise MemoryError('GPU脉压显存不足；请缩短CPI或选择CPU，未静默切换精度') from error
    return output


def align(iq,axes,masks,common,step):
    import torch
    from scipy import fft
    if not np.allclose(np.diff(common),step,rtol=1e-6,atol=1e-15):
        raise ValueError('频域距离重建需要相同采样间隔的公共轴')
    if not np.allclose(np.diff(axes,axis=1),step,rtol=1e-6,atol=1e-15):
        raise ValueError('频域距离重建需要均匀快时间采样')
    output=np.zeros((len(iq),len(common)),dtype=iq.dtype)
    usable=np.count_nonzero(masks,axis=1)>=2
    shifts=(common[0]-axes[:,0])/step
    guards=128+np.ceil(abs(shifts)).astype(int)
    try:
        with torch.inference_mode():
            for guard in np.unique(guards[usable]):
                rows=np.flatnonzero(usable&(guards==guard))
                length=fft.next_fast_len(max(iq.shape[1],len(common))+2*guard)
                frequencies=torch.as_tensor(fft.fftfreq(length),device='cuda:0')
                chunk=_budget(len(rows),length)
                for start in range(0,len(rows),chunk):
                    indices=rows[start:start+chunk]
                    padded=torch.zeros((len(indices),length),dtype=torch.complex128,device='cuda:0')
                    values=np.where(masks[indices],iq[indices],0.).astype(np.complex128)
                    padded[:,guard:guard+iq.shape[1]]=torch.as_tensor(values,device='cuda:0')
                    shift=torch.as_tensor(shifts[indices],device='cuda:0')
                    result=torch.fft.ifft(torch.fft.fft(padded,dim=-1)*torch.exp(2j*np.pi*shift[:,None]*frequencies),dim=-1)
                    output[indices]=result[:,guard:guard+len(common)].cpu().numpy()
    except torch.cuda.OutOfMemoryError as error:
        raise MemoryError('GPU距离对齐显存不足；请缩短CPI或选择CPU') from error
    return output


def slow_fft(cpi,taper):
    import torch
    output=np.empty((len(cpi),cpi.shape[1]),dtype=np.complex128)
    chunk=_budget(cpi.shape[1],len(cpi))
    with torch.inference_mode():
        window=torch.as_tensor(taper,device='cuda:0')[:,None]
        for start in range(0,cpi.shape[1],chunk):
            values=torch.as_tensor(np.array(cpi[:,start:start+chunk],copy=True),device='cuda:0')
            result=torch.fft.fftshift(torch.fft.fft(values*window,dim=0),dim=0)
            output[:,start:start+chunk]=result.cpu().numpy()
    return output.T
