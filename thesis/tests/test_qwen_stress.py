"""CPU-only protocol/controller checks; never claims GH200 measurements."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "thesis/scripts/run_qwen_stress.sh"
embedded = SOURCE.read_text().split("<<'QWEN_STRESS_PY'\n", 1)[1].split("\nQWEN_STRESS_PY\n", 1)[0]
tmp = tempfile.TemporaryDirectory()
driver_path = Path(tmp.name) / "driver.py"
driver_path.write_text(embedded)
spec = importlib.util.spec_from_file_location("stress_driver", driver_path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class StressProtocolTests(unittest.TestCase):
    def test_grid_crosses_128k_and_has_a_finite_cap(self):
        self.assertEqual(driver.coarse_lengths(131072), [1024,2048,4096,8192,16384,32768,65536,131072])
        self.assertEqual(driver.coarse_lengths(1048576)[-3:], [262144,524288,1048576])

    def test_non_power_of_two_cap_is_actually_tested(self):
        self.assertEqual(driver.coarse_lengths(3072), [1024,2048,3072])

    def test_oom_bracket_is_refined_to_1024_without_testing_larger_values(self):
        seen = []
        def trial(t):
            seen.append(t)
            return {"status":"success" if t <= 46080 else "cuda_oom", "stage":"prefill"}
        result = driver.sweep(1048576, True, trial)
        self.assertEqual(result["largest_success_T"], 46080)
        self.assertEqual(result["smallest_capacity_oom_T"], 47104)
        self.assertEqual(result["frontier_resolution_tokens"], 1024)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLessEqual(max(seen), 65536)

    def test_model_load_oom_does_not_establish_a_context_boundary(self):
        result = driver.sweep(131072, True, lambda t:{"status":"cuda_oom","stage":"model_load"})
        self.assertEqual(result["stop_reason"], "cuda_oom")
        self.assertIsNone(result["smallest_capacity_oom_T"])

    def test_timeout_and_other_errors_never_become_oom(self):
        for status in ["error","timeout","process_failed","gpu_busy","interrupted"]:
            with self.subTest(status=status):
                result = driver.sweep(131072, True, lambda t:{"status":status,"stage":"prefill"})
                self.assertEqual(result["stop_reason"], status)
                self.assertIsNone(result["smallest_capacity_oom_T"])

    def test_cap_reached_does_not_claim_oom(self):
        result = driver.sweep(131072, True, lambda t:{"status":"success"})
        self.assertEqual(result["stop_reason"], "max_t_reached_without_oom")
        self.assertEqual(result["largest_success_T"], 131072)
        self.assertIsNone(result["smallest_capacity_oom_T"])

    def test_coarse_only_preserves_the_observed_bracket(self):
        result = driver.sweep(131072, False, lambda t:{"status":"success" if t<=32768 else "cuda_oom","stage":"prefill"})
        self.assertEqual(result["frontier_resolution_tokens"], 32768)

    def test_smallest_input_oom_is_recorded_without_fabricated_success(self):
        result = driver.sweep(131072, True, lambda t:{"status":"cuda_oom","stage":"input_allocation"})
        self.assertIsNone(result["largest_success_T"])
        self.assertEqual(result["smallest_capacity_oom_T"], 1024)

    def test_csv_keeps_integer_bytes_and_success_vs_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            driver.append_result(folder,{"T":1024,"status":"success","peak_allocated_bytes":66045241345})
            driver.append_result(folder,{"T":2048,"status":"cuda_oom","stage":"prefill","error":"Tried to allocate"})
            rows = driver.read_results(folder)
            self.assertEqual(rows[0]["peak_allocated_bytes"], 66045241345)
            self.assertNotIn("peak_allocated_bytes", rows[1])
            self.assertIn("Tried to allocate", (Path(folder)/"results.csv").read_text())

    def test_torn_jsonl_is_not_silently_treated_as_a_complete_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"results.jsonl").write_text('{"T":1024}')
            with self.assertRaisesRegex(RuntimeError,"Truncated"):
                driver.read_results(folder)

    def test_resume_preserves_torn_tail_and_keeps_complete_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"results.jsonl"
            p.write_text('{"T":1024,"status":"success"}\n{"T":2048')
            saved=driver.recover_index_tail(folder)
            self.assertEqual(Path(saved).read_text(),'{"T":2048')
            self.assertEqual(driver.read_results(folder),[{"T":1024,"status":"success"}])

    def test_completed_worker_artifact_can_be_reindexed_without_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=Path(folder)/'T_000001024';trial.mkdir()
            row={'T':1024,'status':'success','protocol_sha256':'test'}
            (trial/'result.json').write_text(json.dumps(row))
            result=driver.run_trial(folder,{'protocol_sha256':'test'},1024,1)
            self.assertEqual(result['status'],'success')
            self.assertTrue(result['recovered_from_worker_artifact'])

    def test_actual_subprocess_timeout_is_bounded_and_not_oom(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fake=root/'fake_worker.py'
            fake.write_text('import time\ntime.sleep(60)\n')
            started=time.monotonic()
            with patch.object(driver,'__file__',str(fake)):
                row=driver.run_trial(root,{'protocol_sha256':'test'},1024,0.1)
            self.assertEqual(row['status'],'timeout')
            self.assertFalse(driver.is_capacity_oom(row))
            self.assertLess(time.monotonic()-started,12)
            self.assertTrue((root/'T_000001024/result.json').exists())

    def test_actual_killed_subprocess_is_not_misclassified_as_cuda_oom(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fake=root/'fake_worker.py'
            fake.write_text('import os,signal\nos.kill(os.getpid(),signal.SIGKILL)\n')
            with patch.object(driver,'__file__',str(fake)):
                row=driver.run_trial(root,{'protocol_sha256':'test'},1024,5)
            self.assertEqual(row['status'],'process_failed')
            self.assertEqual(row['exit_code'],-9)
            self.assertFalse(driver.is_capacity_oom(row))

    def test_gpu_samples_are_target_uuid_only_and_unknown_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"gpu.csv"
            p.write_text('date, 0, GPU-a, 120, 96000\ndate, 1, GPU-b, 900, 96000\ndate, 0, GPU-a, 130, 96000\n')
            self.assertEqual(driver.sampled_gpu_peak(p,"GPU-a"),130)
            self.assertIsNone(driver.sampled_gpu_peak(p,"GPU-c"))

    def test_real_subprocess_import_failure_saves_error_not_oom(self):
        # Dependency absence is guaranteed in an isolated empty stdlib-only subprocess.
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'manifest.json').write_text(json.dumps({"protocol_sha256":"test-protocol","protocol":{}}))
            p=subprocess.run([sys.executable,"-S",str(driver_path),"--worker",str(folder/'manifest.json'),"1024",str(folder)],capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,2)
            row=json.loads((folder/'result.json').read_text())
            self.assertEqual(row['status'],'error')
            self.assertEqual(row['stage'],'imports')
            self.assertEqual(row['protocol_sha256'],'test-protocol')
            self.assertNotIn('peak_allocated_bytes',row)

    def test_baseline_discovery_requires_real_success_revision_and_saved_yarn(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'thesis/outputs/qwen_baseline_001';p.mkdir(parents=True)
            info={"model":driver.MODEL,"B":1,"T":1024,"dtype":"bfloat16","attention":"flash_attention_3","revision":"a"*40}
            config={"model_type":"qwen2","max_position_embeddings":131072,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}
            for name,value in [('model-info.json',info),('model-config.json',config),('result.json',{'status':'success','cache_length':1024})]:
                (p/name).write_text(json.dumps(value))
            self.assertEqual(driver.find_baseline(folder)[0],p)
            config['rope_scaling']['factor']=8.0;(p/'model-config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError,'No successful'):
                driver.find_baseline(folder)


if __name__ == '__main__':
    unittest.main()
