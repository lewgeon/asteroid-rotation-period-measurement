"""Versioned full-grid CPI display products; not a scientific power archive."""
import zipfile
import numpy as np


def display_arrays(power):
    power=np.asarray(power)
    if power.ndim!=2 or not np.all(np.isfinite(power)) or np.any(power<0):
        raise ValueError('CPI功率必须是有限非负二维数组')
    try:from .result_views import _db
    except ImportError:from result_views import _db
    with np.errstate(over='ignore'):marginal=np.sum(power,axis=1,dtype=np.float64)
    if not np.all(np.isfinite(marginal)):raise ValueError('CPI距离功率求和溢出')
    return dict(format_version=np.array(2),power_db=_db(power).astype(np.float32),
        range_power=marginal,power_max=np.array(max(float(np.max(power)),np.finfo(float).tiny)),
        product_kind=np.array('full_grid_display_relative_db'))


def _header(archive,name):
    with archive.open(name+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        shape,order,dtype=np.lib.format._read_array_header(stream,version)
    if dtype.hasobject:raise ValueError('CPI不支持对象数组')
    return shape,dtype


def frame_header(path,byte_budget=512*1024*1024):
    """Validate dimensions before allocating a legacy or compact grid."""
    with zipfile.ZipFile(path) as archive:
        compact='power_db.npy' in archive.namelist()
        shape,dtype=_header(archive,'power_db' if compact else 'power')
        delays,dd=_header(archive,'delays_s');freq,fd=_header(archive,'frequencies_hz')
        if (len(shape)!=2 or len(delays)!=1 or len(freq)!=1 or
                shape!=(delays[0],freq[0]) or not all(shape) or
                dtype.kind!='f' or dd.kind!='f' or fd.kind!='f'):
            raise ValueError('CPI矩阵与坐标维度或类型不一致')
        size=int(shape[0])*int(shape[1])*dtype.itemsize+delays[0]*dd.itemsize+freq[0]*fd.itemsize
        if compact:
            marginal,md=_header(archive,'range_power')
            if marginal!=delays or md!=np.dtype('float64') or dtype!=np.dtype('float32'):
                raise ValueError('显示CPI须使用float32 dB和float64距离功率')
            size+=delays[0]*md.itemsize
        if size>byte_budget:raise ValueError('选定CPI超过读取预算')
    if compact:
        with np.load(path,allow_pickle=False) as data:
            if data['format_version'].shape!=() or data['format_version'].item()!=2 or data['product_kind'].item()!='full_grid_display_relative_db':
                raise ValueError('不支持的CPI显示格式')
    return compact,shape


def read_display(path,byte_budget=512*1024*1024):
    compact,shape=frame_header(path,byte_budget)
    with np.load(path,allow_pickle=False) as data:
        if compact:
            if int(data['format_version'])!=2 or data['product_kind'].item()!='full_grid_display_relative_db':
                raise ValueError('不支持的CPI显示格式')
            db=data['power_db'];marginal=data['range_power'];maximum=float(data['power_max'])
            if not np.all(np.isfinite(db)) or not np.isfinite(maximum) or maximum<=0:
                raise ValueError('CPI显示归一化数据无效')
        else:
            power=data['power']
            # Legacy drawing keeps its original double precision display values.
            try:from .result_views import _db
            except ImportError:from result_views import _db
            if not np.all(np.isfinite(power)) or np.any(power<0):raise ValueError('CPI功率必须有限非负')
            db=_db(power);marginal=power.sum(axis=1,dtype=np.float64)
            maximum=max(float(np.max(power)),np.finfo(float).tiny)
        delays=data['delays_s'];frequencies=data['frequencies_hz']
        if not all(np.all(np.isfinite(x)) for x in (marginal,delays,frequencies)) or np.any(marginal<0):
            raise ValueError('CPI坐标或距离功率无效')
        begin=int(data['begin']);end=int(data['end']);time=float(data['time_s'])
        if begin<0 or end<=begin or not np.isfinite(time):raise ValueError('CPI脉冲范围或时刻无效')
        return dict(power_db=db,range_power=marginal,power_max=maximum,delays_s=delays,
            frequencies_hz=frequencies,time_s=time,begin=begin,end=end,run_id=int(data['run_id']),
            key=data['key'].item(),format_version=2 if compact else 1)
