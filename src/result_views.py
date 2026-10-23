"""Views of saved scientific outputs; never load or process the source IQ."""
from pathlib import Path
import json
import re
import numpy as np


LABELS = {
    'total_power': '总能量', 'doppler_centroid_hz': '多普勒质心（Hz）',
    'centroid_hz': '频谱质心（Hz）',
    'rms_bandwidth_hz': '多普勒 RMS 带宽（Hz）',
    'range_centroid_s': '距离质心（s）', 'centroid_delay_s': '距离质心（s）',
    'energy_width_90_s': '距离 90% 能量宽度（s）',
    'energy_range_width_90_s': '距离 90% 能量宽度（s）',
}


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def feature_view_key(source, config, *, saved=False, echo_npz=None):
    relevant = config if saved else {key: config.get(key) for key in (
        'motion_compensation', 'period_time_role')}
    return dict(version=4, source=file_identity(source), config=relevant, saved=saved,
                echo=(file_identity(echo_npz) if saved and echo_npz is not None and Path(echo_npz).is_file() else None))


def write_html_atomic(figure, path, *, validate=None, standalone=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.html.part')
    try:
        figure.update_layout(height=None,autosize=True)
        include=True
        if not standalone:
            import os,plotly
            from plotly.offline import get_plotlyjs
            # GUI experiments share one local library; normal exports remain
            # standalone and portable. No CDN/network request is introduced.
            state=next((p for p in path.parents if p.name=='.gui_state'),None)
            assets=(state/'previews'/'_assets' if state else path.parent/'_assets')
            assets.mkdir(parents=True,exist_ok=True)
            library=assets/f'plotly-{plotly.__version__}.min.js'
            if not library.is_file():
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=assets,suffix='.part',delete=False) as stream:
                    stream.write(get_plotlyjs());temporary=Path(stream.name)
                try:temporary.replace(library)
                finally:temporary.unlink(missing_ok=True)
            include=Path(os.path.relpath(library,path.parent)).as_posix()
        figure.write_html(str(pending), include_plotlyjs=include,default_height='100vh',default_width='100%',
            config={'responsive':True},post_script="document.body.style.margin='0'; document.body.style.overflow='hidden'; const p=document.getElementById('{plot_id}'); p.style.height='100vh'; p.style.width='100vw'; Plotly.Plots.resize(p); window.addEventListener('resize',()=>Plotly.Plots.resize(p));")
        ensure_responsive_html(pending)
        if validate is not None:
            validate()
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path


def ensure_responsive_html(path):
    """Upgrade historical HTML layout without rerunning its calculation."""
    path=Path(path)
    text=path.read_text(encoding='utf-8')
    original=text
    script="""<style data-viewport-layout>html,body{margin:0;height:100%;overflow:hidden}.plotly-graph-div{width:100vw!important;height:100vh!important}</style><script>function fitResult(){document.querySelectorAll('.plotly-graph-div').forEach(p=>{if(window.Plotly)Plotly.relayout(p,{height:window.innerHeight,width:window.innerWidth,autosize:true});});}window.addEventListener('load',fitResult);window.addEventListener('resize',fitResult);fitResult();</script>"""
    if 'data-viewport-layout' not in text:text=text.replace('</body>',script+'</body>')
    if 'data-run-boundary-toggle="2"' not in text:
        text=re.sub(r'<script data-run-boundary-toggle[^>]*>.*?</script>','',text,flags=re.S)
        toggle="""<script data-run-boundary-toggle="2">
function installRunToggle(){document.querySelectorAll('.plotly-graph-div').forEach(p=>{
if(!window.Plotly||!p.layout||p.dataset.runToggleInstalled==='2')return;
const menus=p.layout.updatemenus||[];
const hide={},show={};
(p.layout.shapes||[]).forEach((s,i)=>{
const legacy=!s.name&&(p.layout.title?.text||'').includes('全观测区间')&&s.type==='line'&&s.xref==='x domain'&&s.x0===0&&s.x1===1&&s.yref==='y'&&s.line?.color==='#96a3b5'&&s.y0===s.y1&&Number.isFinite(s.y0)&&s.y0%1===.5;
if(s.name==='run-boundary'||legacy){hide[`shapes[${i}].visible`]=false;show[`shapes[${i}].visible`]=true;}
});
if(Object.keys(hide).length){
const kept=menus.map(m=>({...m,buttons:(m.buttons||[]).filter(b=>b.label!=='显示/隐藏Run分隔线')})).filter(m=>m.buttons.length);
let group=kept.find(m=>m.buttons.some(b=>b.label==='信号范围'||b.label==='完整记录范围'));
if(!group){group={type:'buttons',buttons:[],y:1.12};kept.unshift(group);}
group.x=0;group.xanchor='left';group.direction='right';group.showactive=false;group.active=-1;
group.buttons.push({label:'显示/隐藏Run分隔线',method:'skip',args:[hide],args2:[show]});
p.dataset.runToggleInstalled='2';
p.on('plotly_buttonclicked',event=>{if(event.button?.label!=='显示/隐藏Run分隔线')return;
const anyVisible=Object.keys(hide).some(key=>p.layout.shapes[Number(key.match(/\\d+/)[0])].visible!==false);
Plotly.relayout(p,anyVisible?hide:show);
});
Plotly.relayout(p,{updatemenus:kept});
}
});}
window.addEventListener('load',installRunToggle);installRunToggle();
</script>"""
        text=text.replace('</body>',toggle+'</body>')
    if text==original:return path
    pending=path.with_suffix('.responsive.part')
    try:
        pending.write_text(text,encoding='utf-8')
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)
    return path


def publish_saved_features(source, config, html_path, echo_npz):
    key = feature_view_key(source, config, saved=True, echo_npz=echo_npz)
    figure = saved_feature_figure(source, current_config=config, current_echo_identity=key['echo'])
    def validate():
        if feature_view_key(source, config, saved=True, echo_npz=echo_npz) != key:
            raise ValueError('生成特征图期间源文件发生变化，请重新查看')
    write_html_atomic(figure, html_path, validate=validate)
    Path(html_path).with_suffix('.view.json').write_text(json.dumps(key), encoding='utf-8')
    return Path(html_path)


def saved_feature_figure(path, current_config=None, current_echo_identity=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    with np.load(path, allow_pickle=False) as data:
        times, values = data['time_elapsed_s'], data['value']
        names = data['feature_names'].tolist()
        runs = data['run_id']
        metadata = json.loads(str(data['metadata_json'].item()))
    if values.shape != (len(times), len(names)) or len(runs) != len(times):
        raise ValueError('保存的特征维度不一致')
    titles = [LABELS.get(name, name) for name in names]
    estimator=metadata.get('inversion_config',{}).get('doppler_estimator','spectral_moments')
    if estimator=='narrowband_lag':
        titles=[title.replace('多普勒 RMS 带宽','低频有效 RMS 宽度') for title in titles]
    figure = make_subplots(rows=len(names), cols=1, shared_xaxes=True,
                          subplot_titles=titles, vertical_spacing=.04)
    custom = np.column_stack((np.arange(len(times)), runs))
    for row, (name, title) in enumerate(zip(names, titles), 1):
        figure.add_trace(go.Scatter(x=times, y=values[:, row - 1], mode='markers',
            name=title, marker=dict(size=4), customdata=custom,
            hovertemplate='观测 %{customdata[0]} / Run %{customdata[1]}<br>时间 %{x:.8g} s<br>值 %{y:.8g}<extra></extra>'), row=row, col=1)
    processing = metadata.get('search_diagnostics', {}).get('processing_path',
        '连续波' if 'centroid_hz' in names else '历史产物')
    saved = metadata.get('inversion_config')
    notice = ''
    if current_config is not None and saved is not None and saved != current_config:
        notice = '；当前参数已修改，显示上次反演快照'
    if saved is None:
        notice += '；历史产物未记录参数快照'
    if metadata.get('echo_identity') is None:
        notice += '；未记录回波身份，无法核对当前回波'
    elif current_echo_identity is None:
        notice += '；当前回波不可核对，显示保存快照'
    if current_echo_identity is not None and metadata.get('echo_identity') is not None and metadata['echo_identity'] != current_echo_identity:
        notice += '；当前回波已变化，显示上次回波的反演快照'
    definition='低频有效滞后拟合' if estimator=='narrowband_lag' else '谱功率矩，含窗展宽'
    if 'doppler_centroid_hz' not in names and 'rms_bandwidth_hz' not in names:
        definition='逐脉冲距离统计，无CPI多普勒'
    figure.update_layout(title=f'反演实际使用的特征：{processing}，{len(times)} 个观测；{definition}{notice}',
                         height=max(600, 230 * len(names)), showlegend=False)
    role = metadata.get('time_axis_role', '未知时标')
    figure.update_xaxes(title_text=f'{role}（相对观测起点，s）', row=len(names), col=1)
    return figure


def save_processing_view(result, output):
    """Export diagnostic samples, retaining the per-frame Doppler frequency axis."""
    output = Path(output)
    dynamic = result.dynamic_spectrum
    power = dynamic.power
    arrays = dict(times_s=dynamic.times_s,
                  metadata_json=np.array(json.dumps(result.search_diagnostics)))
    if power.ndim == 3:
        # All displayed range-time frames, plus three actual slow-time FFT frames.
        indices = np.unique(np.linspace(0, len(power) - 1, min(3, len(power))).astype(int))
        axes = dynamic.frame_frequencies_hz
        frequencies = (np.broadcast_to(dynamic.frequencies_hz, (len(power), len(dynamic.frequencies_hz)))
                       if axes is None else axes)
        arrays.update(range_time_power=power.sum(axis=2), delays_s=dynamic.delays_s,
            rd_power=power[indices], rd_frequencies_hz=frequencies[indices],
            rd_times_s=dynamic.times_s[indices],
            cpi_start_pulse=dynamic.cpi_start_pulse[indices],
            cpi_end_pulse=dynamic.cpi_end_pulse[indices])
    elif hasattr(dynamic, 'delays_s'):
        arrays.update(range_time_power=power, delays_s=dynamic.delays_s)
    else:
        arrays.update(power=power, frequencies_hz=dynamic.frequencies_hz)
    np.savez_compressed(output / 'processing_preview.npz', **arrays)
    # Render from the saved arrays too, so historical diagnostics can be redrawn
    # without loading IQ or repeating FFTs.
    figure = processing_figure(arrays)
    write_html_atomic(figure, output / 'processing.html')


def processing_figure(arrays):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if 'range_time_power' not in arrays:
        figure = go.Figure(go.Heatmap(x=np.arange(len(arrays['times_s'])),
            y=arrays['frequencies_hz'], z=_db(arrays['power']), colorscale='Magma'))
        figure.update_layout(title='连续波动态频谱', xaxis_title='诊断帧序号', yaxis_title='频率（Hz）')
        return figure
    power = arrays['range_time_power']
    delay = arrays['delays_s'] * 1e9
    rd = 'rd_power' in arrays or 'rd_power_db' in arrays
    titles = ['抽样距离像序列：横轴为帧序号，不将观测空档填成信号']
    if rd:
        titles.append('距离—多普勒功率：逐 CPI 查看（抽样诊断，不是完整分辨率 ISAR 图）')
    figure = make_subplots(rows=2 if rd else 1, cols=1, subplot_titles=titles)
    custom = np.broadcast_to(arrays['times_s'], power.T.shape)
    singleton=len(power)==1
    profile_label='所选CPI' if rd else '所选脉冲'
    if singleton:
        figure.add_trace(go.Scatter(x=_db(power[0]),y=delay,mode='lines+markers',marker=dict(size=3),
            customdata=np.full(len(delay),float(arrays['times_s'][0])),
            hovertemplate=profile_label+'时刻 %{customdata:.9g} s<br>延迟 %{y:.6g} ns<br>相对功率 %{x:.4g} dB<extra></extra>'),row=1,col=1)
        figure.layout.annotations[0].text=(profile_label+'的一维距离功率（沿多普勒求和；不是逐脉冲序列）' if rd
            else '所选脉冲的一维距离功率（脉压后；未作慢时间FFT）')
    else:
        figure.add_trace(go.Heatmap(x=np.arange(len(power)), y=delay, z=_db(power).T,
        customdata=custom, zmin=-45, zmax=0, colorscale='Magma', showscale=not rd,
        hovertemplate='诊断帧 %{x}<br>实际时刻 %{customdata:.9g} s<br>延迟 %{y:.6g} ns<br>相对功率 %{z:.4g} dB<extra></extra>'), row=1,col=1)
    marginal = power.sum(axis=0)
    finite = np.flatnonzero(marginal > 0)
    if len(finite):
        cdf=np.cumsum(marginal)/marginal.sum()
        lo,hi=np.searchsorted(cdf,[.005,.995])
        pad=max(2,int((hi-lo)*.15))
        limits=[float(delay[max(0,lo-pad)]),float(delay[min(len(delay)-1,hi+pad)])]
        figure.update_yaxes(range=limits,autorange=False)
    buttons=[]
    if rd:
        frames=arrays.get('rd_power_db',arrays.get('rd_power'))
        for k, frame in enumerate(frames):
            frequencies=arrays['rd_frequencies_hz'][k]
            figure.add_trace(go.Heatmap(x=frequencies,y=delay,z=frame if 'rd_power_db' in arrays else _db(frame),
                zmin=-45,zmax=0,colorscale='Magma',visible=k==0,
                colorbar=dict(title='相对功率 dB')),row=2,col=1)
            label=f"CPI：t={arrays['rd_times_s'][k]:.6g}s；脉冲 [{arrays['cpi_start_pulse'][k]},{arrays['cpi_end_pulse'][k]})"
            buttons.append(dict(label=label,method='update',args=[dict(visible=[True]+[j==k for j in range(len(frames))])]))
        figure.update_xaxes(title_text='多普勒频率（Hz）',row=2,col=1)
    views=[dict(label='聚焦有效延迟',method='relayout',args=[{'yaxis.range':limits,'yaxis2.range':limits,'yaxis.autorange':False,'yaxis2.autorange':False}])] if len(finite) else []
    views.append(dict(label='全部延迟范围',method='relayout',args=[{'yaxis.autorange':True,'yaxis2.autorange':True}]))
    menus=[dict(buttons=views,type='buttons',direction='right',x=0,xanchor='left',y=1.12)]
    if buttons:
        menus.append(dict(buttons=buttons,x=0,y=.48,xanchor='left'))
    step=float(np.median(np.diff(delay))) if len(delay)>1 else 0.
    title=('CPI 可用：真实功率谱' if rd else 'CPI 不可用：没有慢时间 FFT')
    figure.update_layout(title=f'{title}；展示距离格间隔 {step:.6g} ns，默认聚焦中央99%显示能量',
        height=960 if rd else 620, updatemenus=menus, margin=dict(t=130),hovermode='closest',showlegend=False)
    figure.update_xaxes(title_text=profile_label+'的相对距离功率（dB）' if singleton else '诊断帧序号（鼠标查看真实时刻，不表示连续观测）',row=1,col=1)
    if singleton:figure.update_xaxes(range=[-45,0],autorange=False,row=1,col=1)
    figure.update_yaxes(title_text='质心相对延迟（ns）')
    return figure


def render_saved_processing(source, html_path):
    with np.load(source,allow_pickle=False) as data:
        arrays={name:data[name] for name in data.files}
    return write_html_atomic(processing_figure(arrays),html_path)


def _db(power):
    power = np.asarray(power, dtype=float)
    maximum = max(float(np.max(power)), np.finfo(float).tiny)
    return 10 * np.log10(power / maximum + np.finfo(float).tiny)


def processing_times(echo,config):
    role=config.get('period_time_role','scatter_centroid')
    if role in ('scatter_centroid','scatter_elapsed_s'):
        values=echo.scatter_elapsed_s;name='scatter_elapsed_s'
    elif role in ('receive_centroid','receive','receive_elapsed_s'):
        values=echo.elapsed_s;name='receive_elapsed_s'
    else:raise ValueError('不支持的period_time_role')
    if not np.all(np.isfinite(values)):raise ValueError('处理时标必须有限')
    return values,name


def matrix_figure(echo,values,kind,run_id,config,mean_power=None,precomputed_power=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        from .radar_signal import centroid_delay_axes
    except ImportError:
        from radar_signal import centroid_delay_axes
    offsets=centroid_delay_axes(np.array([0.]),len(echo.iq),
        row_fast_time_offset_s=getattr(echo,'row_fast_time_offset_s',None),
        centroid_fractional_offset_s=getattr(echo,'centroid_fractional_offset_s',None))[:,0]
    times,role=processing_times(echo,config)
    delays=echo.fast_time_s+float(np.median(offsets))
    valid=echo.valid[:,None] if echo.valid.ndim==1 else echo.valid
    valid=np.broadcast_to(valid,values.shape)
    power=np.where(valid,np.abs(values)**2 if precomputed_power is None else precomputed_power,0.)
    average=power.mean(axis=0) if mean_power is None else mean_power
    support=np.flatnonzero(average>np.max(average)*1e-8) if np.max(average)>0 else np.arange(len(delays))
    lo=max(0,int(support[0])-8);hi=min(len(delays),int(support[-1])+9)
    step=max(1,int(np.ceil(len(delays)/600)))
    starts=(np.unique(np.r_[np.arange(0,lo,step),np.arange(lo,hi),np.arange(hi,len(delays),step)])
            if hi-lo<=4096 else np.arange(0,len(delays),step))
    stops=np.r_[starts[1:],len(delays)]
    image=np.stack([power[:,left:right].max(axis=1) for left,right in zip(starts,stops)],axis=1)
    observed=np.stack([valid[:,left:right].any(axis=1) for left,right in zip(starts,stops)],axis=1)
    x=np.array([delays[left:right].mean() for left,right in zip(starts,stops)])*1e9
    z=np.where(observed,_db(image),np.nan)
    rows=np.arange(len(values))
    original=getattr(echo,'display_indices',rows)
    custom=np.broadcast_to(np.column_stack((echo.run_id,original,times))[:,None,:],(*z.shape,3))
    figure=make_subplots(rows=2 if kind=='compression' else 1,cols=1,shared_xaxes=True,
        row_heights=[.75,.25] if kind=='compression' else None,vertical_spacing=.12,
        subplot_titles=['脉冲压缩距离像' if kind=='compression' else '原始回波',
                        '平均脉压功率（完整计算）'] if kind=='compression' else ['原始回波'])
    figure.add_trace(go.Heatmap(x=x,y=rows,z=z,customdata=custom,
        colorscale='Magma',zmin=-45,zmax=0,colorbar=dict(title='相对功率dB'),
        hovertemplate='Run %{customdata[0]} / 脉冲 %{customdata[1]}<br>时刻 %{customdata[2]:.9g}s<br>延迟 %{x:.9g}ns<br>%{z:.4g}dB<extra></extra>'),row=1,col=1)
    figure.update_yaxes(title_text='慢时间脉冲顺序（悬停查看真实时刻）',autorange='reversed',row=1,col=1)
    if kind=='compression':
        figure.add_trace(go.Scatter(x=delays*1e9,y=average,name='平均功率'),row=2,col=1)
        figure.update_yaxes(title_text='功率',row=2,col=1)
    limits=[float(delays[lo]*1e9),float(delays[hi-1]*1e9)]
    figure.update_xaxes(title_text='快时间：质心相对延迟（ns）',range=limits)
    updates=[dict(label='信号范围',method='relayout',args=[{'xaxis.range':limits,'xaxis2.range':limits,'xaxis.autorange':False,'xaxis2.autorange':False}]),
        dict(label='完整记录范围',method='relayout',args=[{'xaxis.autorange':True,'xaxis2.autorange':True}])]
    boundaries=np.flatnonzero(np.diff(echo.run_id)!=0)+.5
    for boundary in boundaries:figure.add_hline(y=float(boundary),line_width=1,line_color='#96a3b5',name='run-boundary',row=1,col=1)
    menus=[dict(type='buttons',buttons=updates,direction='right',x=0,xanchor='left',y=1.12,showactive=False,active=-1)]
    if len(boundaries):
        hide={f'shapes[{index}].visible':False for index,shape in enumerate(figure.layout.shapes) if shape.name=='run-boundary'}
        show={key:True for key in hide}
        menus[0]['buttons'].append(dict(label='显示/隐藏Run分隔线',method='skip',args=[hide],args2=[show]))
    scope='全观测区间，Run按顺序排列；灰线为Run边界' if run_id==-1 else f'run_id={run_id}'
    figure.update_layout(title=f"{'脉冲压缩' if kind=='compression' else '原始回波'} · {scope}<br>{role}；原生行显示轴采用中位偏移；仅显示按像素缩减，计算未抽样",
        updatemenus=menus,margin=dict(t=110,b=55),showlegend=False)
    return figure

