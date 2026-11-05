"""CPU-only validation of raw samples, robust summaries and placement gating."""
import csv
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tma_sampled_sweep as sweep


class SampledSweepTests(unittest.TestCase):
    def fixture(self):
        times=[1.,2.,3.,4.,5.,100.]
        permutations=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
        rows=[];raw=[]
        for k,kernel in enumerate(sweep.KERNELS):
            r={field:'0' for field in sweep.FIELDS}
            r.update(input_bytes='4096',tile_rows='32',tile_cols='32',tile_bytes='4096',source='malloc_bound',
                     kernel=kernel,gpu_threads='128',first_sample_position=str(k),kernel_ms='3.5',
                     effective_input_gb_s=str(4096/(3.5*1e6)),status='ok',mbind_success='1',
                     cpu_residency_verified='1',warmup_launches='10',sample_count='6',timing_statistic='median',
                     min_ms='1',p25_ms='2.25',p75_ms='4.75',p95_ms='76.25',max_ms='100',
                     mean_ms=str(115/6),stddev_ms=str(statistics.stdev(times)),
                     cv=str(statistics.stdev(times)/(115/6)),first_sample_ms='1',message='fixture')
            for phase in sweep.PHASES:r.update({'pages_'+phase:'1','on_cpu_'+phase:'1'})
            rows.append(r)
            for i,ms in enumerate(times):
                raw.append(dict(input_bytes='4096',tile_rows='32',tile_cols='32',source='malloc_bound',kernel=kernel,
                                sample_index=str(i+1),position=str(permutations[i].index(k)),kernel_ms=str(ms),
                                effective_input_gb_s=str(4096/(ms*1e6))))
        return rows,raw

    def save(self,folder,rows,raw):
        sweep.write_csv(folder/'rows.csv',sweep.FIELDS,rows)
        sweep.write_csv(folder/'samples.csv',sweep.RAW_FIELDS,raw)

    def test_outlier_is_retained_but_median_is_not_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows,raw=self.fixture();self.save(folder,rows,raw)
            good,samples=sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)
            self.assertEqual(len(samples),18)
            self.assertEqual(good[0]['kernel_ms'],'3.5')
            rows[0]['kernel_ms']='1';self.save(folder,rows,raw)
            with self.assertRaisesRegex(ValueError,'statistic mismatch'):sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)

    def test_missing_or_duplicate_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows,raw=self.fixture();self.save(folder,rows,raw[:-1])
            with self.assertRaises(ValueError):sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)
            self.save(folder,rows,raw+[raw[0]])
            with self.assertRaisesRegex(ValueError,'duplicate raw'):sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)

    def test_order_imbalance_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows,raw=self.fixture()
            for r in raw:r['position']=str(sweep.KERNELS.index(r['kernel']))
            self.save(folder,rows,raw)
            with self.assertRaisesRegex(ValueError,'Unbalanced'):sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)

    def test_post_warmup_migration_blocks_overall_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows,raw=self.fixture();rows[0]['on_cpu_before_samples']='0'
            self.save(folder,rows,raw)
            with self.assertRaisesRegex(ValueError,'Placement flag'):sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)
            rows[0]['cpu_residency_verified']='0';self.save(folder,rows,raw)
            sweep.read_case(folder,(4096,32,32,'malloc_bound'),10,6)
            passed,_=sweep.checkpoint(folder,{(4096,32,32,'malloc_bound'):rows},1,6)
            self.assertFalse(passed)
            self.assertIn('Recorded raw timing samples: 18',(folder/'summary.md').read_text())


if __name__=='__main__':unittest.main()
