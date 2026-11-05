"""CPU-only checks of controlled-sweep validation and independent replication."""
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tma_controlled_sweep as sweep
import test_tma_sampled_sweep as fixtures


class ControlledSweepTests(unittest.TestCase):
    def case(self, folder, rep=0, protocol='gpu_evicted'):
        rows, raw = fixtures.SampledSweepTests().fixture()
        for r in rows:
            k = sweep.sampled.KERNELS.index(r['kernel'])
            r['first_sample_position'] = str((k-rep)%3)
        for r in raw:
            r['position'] = str((sweep.sampled.KERNELS.index(r['kernel'])-rep)%3)
        sweep.sampled.write_csv(folder/'rows.csv', sweep.sampled.FIELDS, rows)
        sweep.sampled.write_csv(folder/'samples.csv', sweep.sampled.RAW_FIELDS, raw)
        c = dict(protocol=protocol, l2_bytes=1024*1024, eviction_bytes=32*512*1024,
                 block_bytes=512*1024, blocks=32, expected_sms=4, minimum_covered_sms=4,
                 passes=2, eviction_calls=18, eviction_kernel_launches=36,
                 primers_per_sample=int(protocol=='self_primed'),
                 hardware_invalidation_claim=False, cpu_cache_flush=False)
        (folder/'conditioning.json').write_text(json.dumps(c))
        return rows, raw, c

    def test_conditioning_and_position_must_match_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows, raw, c = self.case(folder, rep=2)
            sweep.validate_case(folder, (4096,32,32,'malloc_bound'), 'gpu_evicted', 2, 10, 6)
            with self.assertRaisesRegex(ValueError, 'ordering'):
                sweep.validate_case(folder, (4096,32,32,'malloc_bound'), 'gpu_evicted', 0, 10, 6)
            c['minimum_covered_sms'] = 3
            (folder/'conditioning.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'conditioning'):
                sweep.validate_case(folder, (4096,32,32,'malloc_bound'), 'gpu_evicted', 2, 10, 6)

    def test_incomplete_placement_cannot_enter_pooled_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows, raw, c = self.case(folder)
            rows[0]['on_cpu_after'] = '0'
            rows[0]['cpu_residency_verified'] = '0'
            sweep.sampled.write_csv(folder/'rows.csv', sweep.sampled.FIELDS, rows)
            with self.assertRaisesRegex(ValueError, 'placement failed'):
                sweep.validate_case(folder, (4096,32,32,'malloc_bound'), 'gpu_evicted', 0, 10, 6)

    def test_pool_uses_raw_samples_not_median_of_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, raw, c = self.case(Path(tmp))
            cases = []
            for rep in range(3):
                rr, samples = copy.deepcopy(rows), copy.deepcopy(raw)
                times = ([1,1,1,100,100,100] if rep==0 else
                         [2,2,2,2,2,2] if rep==1 else [3,3,3,3,3,3])
                for r in rr:
                    r['kernel_ms'] = str(statistics.median(times))
                for r in samples:
                    r['kernel_ms'] = str(times[int(r['sample_index'])-1])
                cases.append((rr, samples))
            pooled = sweep.pool_cases(cases, 'gpu_evicted')
            # Median of replicate medians is 3; pooled median is 2.5.
            self.assertEqual(pooled[0]['kernel_ms'], 2.5)
            self.assertEqual(pooled[0]['sample_count'], 18)
            self.assertEqual(pooled[0]['replicate_count'], 3)
            self.assertEqual(pooled[0]['pages_after'], 3)
            self.assertGreater(pooled[0]['replicate_median_cv'], 1)

    def test_probe_statistics_and_inconclusive_result_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'probe.json'
            p = dict(self_primed_samples_ms=[1.]*12, gpu_evicted_samples_ms=[2.]*12,
                     self_primed_median_ms=1., gpu_evicted_median_ms=2., evicted_over_primed=2.,
                     probe_separates_conditions=True, minimum_covered_sms=4, expected_sms=4,
                     eviction_bytes=8192, l2_bytes=1024, probe_bytes=256,
                     not_a_hardware_invalidation_proof=True)
            path.write_text(json.dumps(p)); sweep.validate_probe(path)
            p['gpu_evicted_samples_ms'] = [1.]*12
            path.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError, 'statistic mismatch'):
                sweep.validate_probe(path)
            p.update(gpu_evicted_median_ms=1., evicted_over_primed=1.)
            path.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError, 'inconclusive'):
                sweep.validate_probe(path)

    def test_generated_source_preserves_kernels_and_serializer(self):
        root = Path(__file__).resolve().parents[1]/'microbenchmarks/tma_sweep'
        original = (root/'tma_sweep.cu').read_text()
        generated = sweep.controlled_source(*( (root/name).read_text() for name in
            ['tma_sweep.cu','tma_sampled_measure.inc','tma_controlled_conditioning.inc','tma_controlled_measure.inc']))
        start = original.index('__global__ void loads_direct')
        end = original.index('struct Location')
        self.assertTrue(original[start:end] in generated, 'GPU kernel definitions changed')
        self.assertEqual(generated.count('void measure('), 1)
        self.assertEqual(generated.count('void save(const std::string& path,const Memory&'), 1)
        self.assertIn('return controlled_cache_check(argv[2]);', generated)
        self.assertIn('row.positions.push_back(position)', generated)
        self.assertNotIn('sampled_orders[(sample', generated)


if __name__ == '__main__':
    unittest.main()
