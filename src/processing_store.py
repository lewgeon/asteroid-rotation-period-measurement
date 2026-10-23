"""Local preprocessing products shared by inversion and result viewers."""
from pathlib import Path
import json
import zipfile
import shutil
import os
import time
from contextlib import contextmanager
import numpy as np
try:
    from .npz_mapping import try_map_npz_member
except ImportError:
    from npz_mapping import try_map_npz_member


@contextmanager
def stage_lock(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'processing.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        if os.name=='nt':
            import msvcrt
            while True:
                try:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:time.sleep(.1)
            try:yield
            finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def identity(path):
    path=Path(path).resolve();stat=path.stat()
    return dict(path=str(path),size=stat.st_size,mtime=stat.st_mtime_ns)


class MappedArchive:
    def __init__(self,path,folder):
        self.path=Path(path);self.folder=Path(folder)
        self.before=identity(path)
        self.data=np.load(path,allow_pickle=False);self.files=self.data.files
    def __contains__(self,key):return key in self.files
    def __enter__(self):
        self.lock=stage_lock(self.folder);self.lock.__enter__()
        return self
    def __exit__(self,*args):
        try:
            self.data.close()
            if args[0] is None and identity(self.path)!=self.before:
                raise ValueError('读取期间回波发生变化')
        finally:self.lock.__exit__(*args)
    def __getitem__(self,key):
        if key not in ('iq','clean_iq','valid'):return self.data[key]
        mapped=try_map_npz_member(self.path,key,expected_identity=self.before)
        if mapped is not None:return mapped
        self.folder.mkdir(parents=True,exist_ok=True)
        output=self.folder/(key+'.npy');stamp=output.with_suffix('.json')
        try:hit=output.is_file() and json.loads(stamp.read_text(encoding='utf-8'))==self.before
        except (OSError,ValueError):hit=False
        if not hit:
            pending=output.with_suffix('.part')
            try:
                with zipfile.ZipFile(self.path) as archive,archive.open(key+'.npy') as source,pending.open('wb') as destination:
                    shutil.copyfileobj(source,destination,length=1024*1024)
                if identity(self.path)!=self.before:raise ValueError('解压期间回波改变')
                pending.replace(output)
                stamp.write_text(json.dumps(self.before),encoding='utf-8')
            finally:pending.unlink(missing_ok=True)
        return np.load(output,mmap_mode='r',allow_pickle=False)


def stage_key(echo,config):
    policy={key:config.get(key) for key in ('motion_compensation','period_time_role','cpi_duration_s','cpi_hop_duration_s','doppler_estimator')}
    policy['doppler_estimator']=config.get('doppler_estimator','spectral_moments')
    policy['motion_compensation']=config.get('motion_compensation','auto')
    policy['period_time_role']=config.get('period_time_role','scatter_centroid')
    if config.get('compute_device','cpu')!='cpu':
        try:from .compute_backend import resolve_device
        except ImportError:from compute_backend import resolve_device
        policy['compute_execution']=dict(resolve_device(config),backend_version=1)
    # v4 uses full-correlation energy for every CPI range statistic and saves
    # spectral-window diagnostics. Old scalar observations must be recomputed.
    key=dict(version=4,source=echo._source_identity,policy=policy)
    if config.get('export_full_cpi_power',False):
        key['output_policy']={'export_full_cpi_power':True}
    return key


def cpi_key(echo,config):
    """Identity of the spectrum, independent of the feature estimator/output."""
    policy=_spectrum_policy(stage_key(echo,config)['policy'])
    return dict(version=1,algorithm_version=3,source=echo._source_identity,policy=policy)


def _spectrum_policy(policy):
    policy=dict(policy)
    policy.pop('doppler_estimator',None)
    execution=policy.get('compute_execution',{})
    policy['compute_execution']=dict(actual=execution.get('actual','cpu'),
        backend_version=execution.get('backend_version',1))
    return policy


def cpi_key_matches(key,echo,config):
    expected=cpi_key(echo,config)
    if key==expected:return True
    # v3 archives predate independent spectrum identities. Only their feature
    # selector/output policy may differ; every spectrum dependency must match.
    if not isinstance(key,dict) or key.get('version') not in (3, 4):return False
    if not isinstance(key.get('policy'),dict):return False
    execution=key['policy'].get('compute_execution',{})
    if not isinstance(execution,dict):return False
    policy=_spectrum_policy(key.get('policy',{}))
    return key.get('source')==expected['source'] and policy==expected['policy']


def attach_store(echo,source):
    echo._source_identity=identity(source)
    echo._processing_store=Path(source).parent/'processing_cache'
    return echo


def save_cpi(echo,config,cube,number,begin,end):
    try:from .cpi_display import display_arrays
    except ImportError:from cpi_display import display_arrays
    folder=cpi_folder(echo,config);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{number}.npz';pending=path.with_suffix('.pending.npz')
    # Spectra are independent of the feature estimator. Do not rewrite a valid
    # shared frame when only the estimator changes.
    try:
        with np.load(path,allow_pickle=False) as data:
            if (cpi_key_matches(json.loads(data['key'].item()),echo,config) and
                    int(data['begin'])==begin and int(data['end'])==end and
                    int(data['run_id'])==int(np.asarray(echo.run_id)[begin]) and
                    (not config.get('export_full_cpi_power',False) or 'power' in data.files)):
                try:from .cpi_display import read_display
                except ImportError:from cpi_display import read_display
                read_display(path)
                return
    except (OSError,ValueError,KeyError,zipfile.BadZipFile):pass
    try:
        power=cube.power[0];values=display_arrays(power)
        if config.get('export_full_cpi_power',False):values['power']=power
        np.savez_compressed(pending,**values,delays_s=cube.delays_s,
            frequencies_hz=cube.frame_frequencies_hz[0],time_s=cube.times_s[0],
            begin=begin,end=end,run_id=np.asarray(echo.run_id)[begin],
            key=np.array(json.dumps(cpi_key(echo,config))))
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def cpi_folder(echo,config):
    key=cpi_key(echo,config)
    return _cpi_policy_folder(echo,key['policy'],prefix=f"s{key['version']}_a{key['algorithm_version']}_")


def _cpi_policy_folder(echo,policy,prefix=''):
    values=[]
    for key,value in policy.items():
        if key=='compute_execution':
            requested=(value['requested'].replace(':','-')+'_' if 'requested' in value else '')
            values.append(f"{requested}{value['actual'].replace(':','-')}_b{value['backend_version']}")
        else:values.append(str(value))
    name=prefix+'_'.join(values)+f"_{echo._source_identity['mtime']}_{echo._source_identity['size']}"
    return Path(echo._processing_store)/'cpi'/name


def cpi_folder_candidates(echo,config):
    """Current spectrum directory first, then both legacy estimator paths."""
    folders=[cpi_folder(echo,config)]
    estimators=(config.get('doppler_estimator','spectral_moments'),'spectral_moments','narrowband_lag')
    for estimator in estimators:
        legacy=dict(config,doppler_estimator=estimator)
        path=_cpi_policy_folder(echo,stage_key(echo,legacy)['policy'])
        if path not in folders:folders.append(path)
        if cpi_key(echo,config)['policy']['compute_execution']['actual']=='cpu':
            path=_cpi_policy_folder(echo,stage_key(echo,dict(legacy,compute_device='cpu'))['policy'])
            if path not in folders:folders.append(path)
    return folders


def save_stage(echo,config,stage):
    from dataclasses import is_dataclass,fields
    folder=Path(echo._processing_store);folder.mkdir(parents=True,exist_ok=True)
    values={};description={};objects={};arrays={}
    def array_reference(value,name):
        # Object identity is deliberate: equal independent arrays and views
        # remain independent. No scan/hash of scientific arrays is required.
        if id(value) not in arrays:
            arrays[id(value)]=name;values[name]=value
        return dict(array=arrays[id(value)])
    for name,value in stage.items():
        if is_dataclass(value):
            if id(value) in objects:
                description[name]=dict(ref=objects[id(value)]);continue
            objects[id(value)]=name
            description[name]=dict(type=type(value).__name__,fields={})
            for item in fields(value):
                part=getattr(value,item.name)
                description[name]['fields'][item.name]=(array_reference(part,name+'__'+item.name)
                    if part is not None else False)
        elif isinstance(value,np.ndarray):description[name]=array_reference(value,name)
        else:description[name]=dict(value=value)
    values['manifest']=np.array(json.dumps(dict(key=stage_key(echo,config),description=description)))
    path=folder/'observations.npz';pending=folder/'observations.pending.npz'
    try:
        np.savez_compressed(pending,**values)
        if identity(echo._source_identity['path'])!=echo._source_identity:raise ValueError('处理中回波改变')
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def load_stage(echo,config):
    path=Path(echo._processing_store)/'observations.npz'
    if identity(echo._source_identity['path'])!=echo._source_identity:
        raise ValueError('回波已变化，请重新加载回波')
    try:
        try:from . import radar_signal as signal
        except ImportError:import radar_signal as signal
        with np.load(path,allow_pickle=False) as data:
            manifest=json.loads(data['manifest'].item())
            if manifest['key']!=stage_key(echo,config):return None
            result={};arrays={};active=set()
            def array(name):
                if name not in arrays:arrays[name]=data[name]
                return arrays[name]
            def restore(name):
                if name in result:return result[name]
                if name in active:raise ValueError('循环阶段引用')
                active.add(name)
                description=manifest['description'][name]
                if description=='array':value=array(name)
                elif 'ref' in description:value=restore(description['ref'])
                elif 'array' in description:value=array(description['array'])
                elif 'type' in description:
                    cls=getattr(signal,description['type'])
                    value=cls(**{key:(array(exists['array']) if isinstance(exists,dict)
                        else array(name+'__'+key) if exists else None)
                        for key,exists in description['fields'].items()})
                else:value=description['value']
                result[name]=value;active.remove(name)
                return value
            for name in manifest['description']:restore(name)
            return result
    except (OSError,ValueError,KeyError,AttributeError,TypeError,IndexError,zipfile.BadZipFile):return None
