"""Saved inspection data produced only by the inversion execution entry point."""
import json,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
try:
    from . import inversion as core,result_views
    from .processing_store import identity,stage_lock
    from .compute_backend import configured,resolve_device
except ImportError:
    import inversion as core,result_views
    from processing_store import identity,stage_lock
    from compute_backend import configured,resolve_device
    if not hasattr(core,'prepare_chirp_pulse_profiles'):
        import importlib.util,sys
        spec=importlib.util.spec_from_file_location('saved_inspection_core',Path(__file__).with_name('inversion.py'))
        core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)


def inspection_key(source,config):
    key=dict(version=1,source=identity(source),policy=dict(
        motion_compensation=config.get('motion_compensation','auto'),
        period_time_role=config.get('period_time_role','scatter_centroid')))
    if config.get('compute_device','cpu')!='cpu':key['policy']['compute_execution']=dict(resolve_device(config),backend_version=1)
    return key


def inspection_folder(source,config):
    key=inspection_key(source,config);p=key['policy'];s=key['source']
    from urllib.parse import quote
    name=quote(f"{p['motion_compensation']}_{p['period_time_role']}_{s['mtime']}_{s['size']}",safe='')
    if 'compute_execution' in p:name+='_'+p['compute_execution']['requested'].replace(':','-')+'_'+p['compute_execution']['actual'].replace(':','-')
    return Path(source).parent/'processing_cache'/'inspection'/name


def ready_inspection(source,config):
    folder=inspection_folder(source,config)
    try:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['key']!=inspection_key(source,config):return None
        required={'pulse_observables.npz','compression_campaign.npz'}|{f'compression_run_{int(run)}.npz' for run in manifest['runs']}
        if set(manifest['artifacts'])!=required:return None
        for name in required:
            if result_views.file_identity(folder/name)!=manifest['artifacts'][name]:return None
        return folder
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return None


def write_inspections(echo,source,config,progress=None):
    with stage_lock(inspection_folder(source,config)):
        return _write_inspections(echo,source,config,progress)


def _write_inspections(echo,source,config,progress=None):
    expected=getattr(echo,'_source_identity',None)
    if expected is None or identity(source)!=expected:
        raise ValueError('检查数据的已加载回波身份不匹配，不能认证新来源')
    if np.ndim(echo.iq)!=2:return None
    folder=ready_inspection(source,config)
    if folder is not None:return folder
    folder=inspection_folder(source,config);folder.mkdir(parents=True,exist_ok=True)
    key=inspection_key(source,config);n=len(echo.iq)
    source_echo=core._pulse_feature_source(echo,n)
    runs=np.asarray(source_echo.run_id);run_values=np.unique(runs)
    shown_runs=set(run_values[np.unique(np.linspace(0,len(run_values)-1,min(1000,len(run_values)),dtype=int))])
    chunk_size=min(128,core.bounded_pulse_chunk_size(source_echo,32*1024*1024))
    parts={name:np.full(n,np.nan) for name in ('times_s','total_power','centroid_delay_s','energy_width_90_s')}
    display=[];display_power=[];total=np.zeros(echo.iq.shape[1]);role=None
    def selected(indices):
        chunk=copy.copy(source_echo)
        for name in core._PULSE_ROW_FIELDS:
            value=getattr(chunk,name,None)
            if value is not None:setattr(chunk,name,value[indices])
        return chunk
    def figure_for(indices,power,mean,run):
        shown=selected(indices);shown.iq=power;shown.display_indices=indices
        return result_views.matrix_figure(shown,power,'compression',int(run),config,
            mean_power=mean,precomputed_power=power)
    for number,run in enumerate(run_values):
        indices=np.flatnonzero(runs==run)
        wanted=np.unique(np.linspace(0,len(indices)-1,min(1000,len(indices)),dtype=int))
        campaign=(np.unique(np.linspace(0,len(indices)-1,min(len(indices),max(1,1000//len(shown_runs))),dtype=int))
            if run in shown_runs else np.array([],dtype=int))
        retained=[];sum_power=np.zeros(echo.iq.shape[1])
        for begin in range(0,len(indices),chunk_size):
            ids=indices[begin:begin+chunk_size];chunk=selected(ids)
            prepared=core.prepare_chirp_pulse_profiles(chunk,config,pulse_features_only=True)
            feature=core._native_pulse_range_features(prepared)
            for name in parts:parts[name][ids]=getattr(feature,name)
            role=prepared['time_axis_role'];lag=prepared['full_compressed'].shape[1]-echo.iq.shape[1]
            compressed=prepared['full_compressed'][:,lag:lag+echo.iq.shape[1]]
            valid=chunk.valid[:,None] if chunk.valid.ndim==1 else chunk.valid
            power=np.where(valid,np.abs(compressed)**2,0.)
            sum_power+=power.sum(axis=0)
            chosen=wanted[(wanted>=begin)&(wanted<begin+len(ids))]-begin
            retained.append(power[chosen].copy())
            local=campaign[(campaign>=begin)&(campaign<begin+len(ids))]-begin
            if len(local):display.extend(ids[local]);display_power.append(power[local].copy())
            del prepared,compressed,power,chunk
        total+=sum_power
        figure=figure_for(indices[wanted],np.concatenate(retained),sum_power/len(indices),run)
        pending=folder/f'compression_run_{int(run)}.pending.npz'
        save_compression_figure(pending,figure,key)
        pending.replace(folder/f'compression_run_{int(run)}.npz')
        if progress:progress('inversion',90+int(4*(number+1)/len(run_values)),f'保存逐脉冲和脉压检查 {number+1}/{len(run_values)} Run')
    figure=figure_for(np.asarray(display),np.concatenate(display_power),total/n,-1)
    save_compression_figure(folder/'compression_campaign.npz',figure,key)
    np.savez_compressed(folder/'pulse_observables.npz',time_elapsed_s=parts['times_s'],
        value=np.column_stack([parts[name] for name in ('total_power','centroid_delay_s','energy_width_90_s')]),
        feature_names=['total_power','centroid_delay_s','energy_width_90_s'],run_id=runs,
        metadata_json=np.array(json.dumps(dict(inversion_config=config,time_axis_role=role,
            echo_identity=result_views.file_identity(source),search_diagnostics=dict(processing_path='pulse_inspection')))))
    if inspection_key(source,config)!=key:raise ValueError('保存检查结果期间回波改变')
    names=['pulse_observables.npz','compression_campaign.npz']+[f'compression_run_{int(run)}.npz' for run in run_values]
    artifacts={name:result_views.file_identity(folder/name) for name in names}
    pending=folder/'manifest.pending.json';pending.write_text(json.dumps(dict(key=key,runs=run_values.tolist(),artifacts=artifacts)),encoding='utf-8')
    pending.replace(folder/'manifest.json')
    return folder


def compression_figure(source,config,run):
    folder=ready_inspection(source,config)
    if folder is None:raise ValueError('没有保存的脉压检查结果；请运行周期反演')
    path=folder/('compression_campaign.npz' if run==-1 else f'compression_run_{int(run)}.npz')
    with np.load(path,allow_pickle=False) as data:
        if json.loads(data['key'].item())!=inspection_key(source,config):raise ValueError('脉压检查来源已变化')
        return load_compression_figure(data)


def save_compression_figure(path,figure,key):
    """Keep display grids and row metadata out of repeated pixel JSON."""
    values={};restore={};cleared=[]
    try:
        for index,trace in enumerate(figure.data):
            if trace.type!='heatmap' or trace.z is None:continue
            z=np.asarray(trace.z);custom=np.asarray(trace.customdata)
            if z.ndim!=2:raise ValueError('检查图必须使用二维显示网格')
            values[f'z_{index}']=z.astype(np.float32)
            record=dict(z=f'z_{index}')
            if custom.ndim==3 and custom.shape[:2]==z.shape:
                # matrix_figure creates row-constant run/pulse/time metadata.
                if not np.array_equal(custom,np.broadcast_to(custom[:,0:1,:],custom.shape),equal_nan=True):
                    raise ValueError('检查图元数据必须逐行共享')
                values[f'rows_{index}']=custom[:,0,:]
                record['rows']=f'rows_{index}'
            elif trace.customdata is not None:
                raise ValueError('不支持的检查图元数据结构')
            cleared.append((trace,trace.z,trace.customdata))
            trace.z=None;trace.customdata=None;restore[str(index)]=record
        values.update(format_version=np.array(2),restore_json=np.array(json.dumps(restore)),
            figure_json=np.array(figure.to_json()),key=np.array(json.dumps(key)))
        np.savez_compressed(path,**values)
    finally:
        for trace,z,custom in cleared:trace.z=z;trace.customdata=custom


def load_compression_figure(data):
    import plotly.io as pio
    figure=pio.from_json(data['figure_json'].item())
    if 'format_version' not in data.files:return figure
    if int(data['format_version'])!=2:raise ValueError('不支持的脉压检查格式')
    for index,record in json.loads(data['restore_json'].item()).items():
        trace=figure.data[int(index)];z=data[record['z']]
        if z.ndim!=2 or z.dtype!=np.float32:raise ValueError('检查显示网格维度或类型不正确')
        trace.z=z
        if 'rows' in record:
            rows=data[record['rows']]
            if rows.ndim!=2 or len(rows)!=len(z):raise ValueError('检查图逐行信息长度不一致')
            trace.customdata=np.broadcast_to(rows[:,None,:],(*z.shape,rows.shape[1]))
    return figure
