"""CPU-only checks for grid coverage, residency gating, and checkpoint reuse."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import run_malloc_bound_sweep as sweep


class BoundSweepTests(unittest.TestCase):
    def rows(self,config=(4096,32,32)):
        n,h,w=config
        records=[]
        for order,kernel in enumerate(sorted(sweep.KERNELS)):
            r=dict(input_bytes=str(n),tile_rows=str(h),tile_cols=str(w),tile_bytes=str(h*w*4),
                   source='malloc_bound',kernel=kernel,gpu_threads='128',launch_order=str(order),
                   kernel_ms='1',effective_input_gb_s=str(n/1e6),status='ok',cpu_node='0',
                   mbind_success='1',cpu_residency_verified='1',message='')
            for phase in ['before','after','after_verification']:
                r.update({f'pages_{phase}':'1',f'on_cpu_{phase}':'1',f'unknown_{phase}':'0',f'query_errno_{phase}':'0'})
            records.append(r)
        return records

    def save(self,path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_grid_covers_original_valid_pairs(self):
        grid=sweep.configurations()
        self.assertEqual(len(grid),145)
        self.assertEqual(len(set(grid)),145)
        self.assertEqual([g for g in grid if g[0]==4096],[(4096,32,32)])
        self.assertEqual(len([g for g in grid if g[0]==16<<30]),7)

    def test_late_migration_cannot_be_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);config=(4096,32,32)
            folder=out/'cases/4096_32x32';path=folder/'rows.csv'
            rows=self.rows();rows[0]['on_cpu_after_verification']='0'
            self.save(path,rows)
            with self.assertRaisesRegex(ValueError,'placement/binding'):
                sweep.read_case(path,config)
            rows[0]['cpu_residency_verified']='0';self.save(path,rows)
            (folder/'done.json').write_text('{"returncode":0}')
            passed,records=sweep.collect(out,[config])
            self.assertFalse(passed)
            self.assertEqual(len(records),3)
            self.assertIn('2 / 3',(out/'summary.md').read_text())

    def test_checkpoint_ignores_unfinished_case_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);config=(4096,32,32);folder=out/'cases/4096_32x32';path=folder/'rows.csv'
            rows=self.rows();self.save(path,rows)
            self.assertEqual(sweep.collect(out,[config])[1],[])
            (folder/'done.json').write_text('{"returncode":0}')
            self.assertTrue(sweep.collect(out,[config])[0])
            rows[1]['kernel']=rows[0]['kernel'];self.save(path,rows)
            with self.assertRaises(ValueError):sweep.collect(out,[config])

if __name__=='__main__':unittest.main()
