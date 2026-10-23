"""Storage identity contracts: aliasing is structural, never content based."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inversion.src import processing_store as store
from inversion.src.radar_signal import RangeProfiles, RangeDopplerCube


class CompactStageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        source=Path(self.temp.name)/'echo.npz'
        source.write_bytes(b'fixed scientific source')
        self.echo=store.attach_store(SimpleNamespace(),source)

    def test_shared_dataclass_and_array_round_trip(self):
        times=np.array([.25,.75,1.25])
        iq=np.array([[1+2j,3-4j],[5+6j,7-8j],[9+10j,11-12j]])
        dynamic=RangeProfiles(times,np.array([2.,5.]),iq)
        store.save_stage(self.echo,{},dict(dynamic=dynamic,profiles=dynamic,
            feature_times=times,other_times=times,empty=None,notices=['example']))
        with np.load(Path(self.echo._processing_store)/'observations.npz') as data:
            self.assertEqual(set(data.files),{'dynamic__times_s','dynamic__delays_s',
                'dynamic__compressed_iq','manifest'})
            manifest=json.loads(data['manifest'].item())
            self.assertEqual(manifest['description']['profiles'],{'ref':'dynamic'})
        loaded=store.load_stage(self.echo,{})
        self.assertIsNotNone(loaded)
        self.assertIs(loaded['profiles'],loaded['dynamic'])
        self.assertIs(loaded['feature_times'],loaded['dynamic'].times_s)
        self.assertIs(loaded['feature_times'],loaded['other_times'])
        np.testing.assert_array_equal(loaded['dynamic'].compressed_iq,iq)
        self.assertEqual(loaded['dynamic'].compressed_iq.dtype,iq.dtype)
        self.assertIsNone(loaded['empty'])
        self.assertEqual(loaded['notices'],['example'])

    def test_equal_but_distinct_objects_and_views_not_aliased(self):
        iq=np.arange(6,dtype=np.float64).reshape(3,2)+1j
        first=RangeProfiles(np.arange(3.),np.arange(2.),iq)
        second=RangeProfiles(first.times_s.copy(),first.delays_s.copy(),iq.copy())
        view=iq.view()
        # Guard the fixture: equal values deliberately have distinct identity.
        np.testing.assert_array_equal(iq,second.compressed_iq)
        self.assertIsNot(iq,second.compressed_iq)
        self.assertTrue(np.shares_memory(iq,view))
        store.save_stage(self.echo,{},dict(first=first,second=second,view=view))
        loaded=store.load_stage(self.echo,{})
        self.assertIsNotNone(loaded)
        self.assertIsNot(loaded['first'],loaded['second'])
        self.assertIsNot(loaded['first'].compressed_iq,loaded['second'].compressed_iq)
        self.assertIsNot(loaded['first'].compressed_iq,loaded['view'])
        loaded['first'].compressed_iq[0,0]=123
        self.assertNotEqual(loaded['first'].compressed_iq[0,0],loaded['second'].compressed_iq[0,0])
        self.assertNotEqual(loaded['first'].compressed_iq[0,0],loaded['view'][0,0])

    def test_optional_cube_arrays_and_legacy_manifest(self):
        cube=RangeDopplerCube(np.array([.1]),np.arange(2.),np.arange(3.),
            np.arange(6).reshape(1,2,3).astype(complex),np.array([2]),np.array([0]),np.array([12]))
        store.save_stage(self.echo,{},dict(dynamic=cube,profiles=cube))
        loaded=store.load_stage(self.echo,{})
        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded['dynamic'].energy_range_power)
        np.testing.assert_array_equal(loaded['dynamic'].spectrum,cube.spectrum)
        # Reproduce the pre-change bool-field manifest independently.
        folder=Path(self.echo._processing_store)
        description={'profiles':{'type':'RangeProfiles','fields':{
            'times_s':True,'delays_s':True,'compressed_iq':True}},
            'array':'array','role':{'value':'scatter_elapsed_s'}}
        np.savez_compressed(folder/'observations.npz',profiles__times_s=[.2,.6],
            profiles__delays_s=[3.,8.],profiles__compressed_iq=[[1+2j,3+4j],[5+6j,7+8j]],
            array=[9,11],manifest=np.array(json.dumps({'key':store.stage_key(self.echo,{}),
                'description':description})))
        loaded=store.load_stage(self.echo,{})
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded['profiles'].compressed_iq,[[1+2j,3+4j],[5+6j,7+8j]])
        np.testing.assert_array_equal(loaded['array'],[9,11])
        self.assertEqual(loaded['role'],'scatter_elapsed_s')

    def test_feature_estimator_and_explicit_output_invalidate_stage(self):
        store.save_stage(self.echo,{},dict(array=np.arange(5)))
        self.assertIsNone(store.load_stage(self.echo,{'doppler_estimator':'narrowband_lag'}))
        self.assertIsNone(store.load_stage(self.echo,{'export_full_cpi_power':True}))
        self.assertIsNotNone(store.load_stage(self.echo,{'export_full_cpi_power':False}))

    def test_array_first_and_separate_dataclasses_share_only_same_arrays(self):
        shared=np.array([.3,.7])
        a=RangeProfiles(shared,np.array([2.,6.]),np.array([[1j,2j],[3j,4j]]))
        b=RangeProfiles(shared,np.array([2.,6.]),a.compressed_iq.copy())
        store.save_stage(self.echo,{},dict(times=shared,a=a,b=b))
        loaded=store.load_stage(self.echo,{})
        self.assertIsNotNone(loaded)
        self.assertIs(loaded['times'],loaded['a'].times_s)
        self.assertIs(loaded['a'].times_s,loaded['b'].times_s)
        self.assertIsNot(loaded['a'],loaded['b'])
        self.assertIsNot(loaded['a'].delays_s,loaded['b'].delays_s)
        self.assertIsNot(loaded['a'].compressed_iq,loaded['b'].compressed_iq)

    def test_broken_references_are_cache_misses(self):
        store.save_stage(self.echo,{},dict(a=np.arange(3)))
        path=Path(self.echo._processing_store)/'observations.npz'
        for description in ({'a':{'ref':'missing'}},
                {'a':{'ref':'b'},'b':{'ref':'a'}},
                {'a':{'array':'missing'}}):
            with self.subTest(description=description):
                np.savez_compressed(path,manifest=np.array(json.dumps({
                    'key':store.stage_key(self.echo,{}),'description':description})))
                self.assertIsNone(store.load_stage(self.echo,{}))

    def test_cpi_identity_dependency_matrix(self):
        baseline={'cpi_duration_s':12.,'cpi_hop_duration_s':3.}
        original=store.cpi_key(self.echo,baseline)
        irrelevant=[{'doppler_estimator':'narrowband_lag'},
            {'export_full_cpi_power':True},{'uncertainty_method':'run_bootstrap'}]
        for change in irrelevant:
            with self.subTest(change=change):
                config=dict(baseline,**change)
                self.assertEqual(original,store.cpi_key(self.echo,config))
                self.assertEqual(store.cpi_folder(self.echo,baseline),store.cpi_folder(self.echo,config))
        for change in ({'motion_compensation':'none'},{'period_time_role':'receive'},
                       {'cpi_duration_s':24.},{'cpi_hop_duration_s':6.}):
            with self.subTest(change=change):
                self.assertNotEqual(original,store.cpi_key(self.echo,dict(baseline,**change)))
                self.assertFalse(store.cpi_key_matches(original,self.echo,dict(baseline,**change)))
        for field,value in (('path','other_source'),('size',500),('mtime',700)):
            other=SimpleNamespace(_source_identity=dict(self.echo._source_identity,**{field:value}))
            self.assertNotEqual(original,store.cpi_key(other,baseline))
        changed=dict(original,algorithm_version=original['algorithm_version']+1)
        self.assertFalse(store.cpi_key_matches(changed,self.echo,baseline))

    def test_actual_backend_identity_and_legacy_paths(self):
        config={'compute_device':'cuda:0','doppler_estimator':'narrowband_lag'}
        with patch('inversion.src.compute_backend.resolve_device',return_value={
                'requested':'cuda:0','actual':'cpu','reason':'fallback'}):
            cpu=store.cpi_key(self.echo,config)
            self.assertEqual(cpu,store.cpi_key(self.echo,{}))
            self.assertEqual(store.cpi_folder(self.echo,config),store.cpi_folder(self.echo,{}))
            self.assertNotEqual(store.stage_key(self.echo,config),store.stage_key(self.echo,{}))
            old_fallback=store.stage_key(self.echo,config)
            self.assertTrue(store.cpi_key_matches(old_fallback,self.echo,{}))
            self.assertTrue(store.cpi_key_matches(store.stage_key(self.echo,{}),self.echo,config))
            legacy_paths=store.cpi_folder_candidates(self.echo,config)
            self.assertTrue(any('cuda-0_cpu_b1' in path.name for path in legacy_paths[1:]))
            newer_backend=json.loads(json.dumps(old_fallback))
            newer_backend['policy']['compute_execution']['backend_version']=2
            self.assertFalse(store.cpi_key_matches(newer_backend,self.echo,config))
        with patch('inversion.src.compute_backend.resolve_device',return_value={
                'requested':'cuda:0','actual':'cuda:0','reason':None}):
            gpu=store.cpi_key(self.echo,config)
            self.assertNotEqual(cpu,gpu)
            self.assertNotIn(':',store.cpi_folder(self.echo,config).name)
        config={'doppler_estimator':'narrowband_lag','export_full_cpi_power':True}
        legacy=store.stage_key(self.echo,{'doppler_estimator':'spectral_moments'})
        self.assertTrue(store.cpi_key_matches(legacy,self.echo,config))
        paths=store.cpi_folder_candidates(self.echo,config)
        self.assertEqual(paths[0],store.cpi_folder(self.echo,config))
        self.assertEqual(len(paths),3)
        suffix=f"_{self.echo._source_identity['mtime']}_{self.echo._source_identity['size']}"
        self.assertEqual(paths[1].name,'auto_scatter_centroid_None_None_narrowband_lag'+suffix)
        self.assertEqual(paths[2].name,'auto_scatter_centroid_None_None_spectral_moments'+suffix)


if __name__=='__main__':unittest.main()
