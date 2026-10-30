"""Ελέγχει τον Python runner χωρίς GPU. Δεν ελέγχει τη λειτουργία του TMA."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'microbenchmarks/tma_cpu/run.py'
spec = importlib.util.spec_from_file_location('tma_cpu_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_nonzero_cannot_claim_success(self):
        # Ένα μήνυμα επιτυχίας στο log δεν αρκεί αν η διεργασία έβγαλε σφάλμα.
        row = runner.parse_worker('STAGE correctness\nRESULT {"status":"correctness_pass"}\n', 2)
        self.assertEqual(row['status'], 'worker_error')
        self.assertEqual(row['stage'], 'correctness')

    def test_missing_artifact_is_failure(self):
        # Η διεργασία πρέπει να έχει γράψει και αποτέλεσμα, όχι μόνο να έχει τερματίσει.
        self.assertEqual(runner.parse_worker('STAGE allocation\n', 0)['status'], 'worker_error')

    def test_residency_and_control_required_for_verdict(self):
        # Σωστή αντιγραφή και επιβεβαιωμένη θέση σελίδων πρέπει να αναφέρονται χωριστά.
        row = lambda source, method, resident: dict(source=source, method=method,
            status='correctness_pass', cpu_residency_verified_before_and_after=resident)
        rows = [row('hbm', 'tma', False), row('pinned', 'tma', False), row('registered', 'tma', True)]
        self.assertIn('check_per_allocation_residency', runner.verdict(rows))
        rows[1]['cpu_residency_verified_before_and_after'] = True
        self.assertIn('residency_observed', runner.verdict(rows))
        rows[0]['status'] = 'worker_error'
        self.assertIn('no_host_tma_conclusion', runner.verdict(rows))

    def run_fixture(self, program, timeout=2):
        # Χρησιμοποιούμε μικρό Python πρόγραμμα στη θέση του CUDA εκτελέσιμου.
        # Έτσι ελέγχουμε πώς χειρίζεται ο runner τα σφάλματα, χωρίς να ανοίξουμε GPU context.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            binary = folder / 'fixture'
            binary.write_text('#!/usr/bin/env python3\n' + program)
            binary.chmod(0o700)
            row = runner.supervise(binary, 'pinned', 'tma', 4096, 3, folder/'trial', timeout)
            self.assertEqual(row, json.loads((folder/'trial/result.json').read_text()))
            return row

    def test_real_child_error_is_saved(self):
        row = self.run_fixture('print("STAGE tensor_map_encode", flush=True)\nraise SystemExit(2)\n')
        self.assertEqual(row['stage'], 'tensor_map_encode')
        self.assertEqual(row['exit_code'], 2)

    def test_real_timeout_is_not_unsupported(self):
        row = self.run_fixture('import time\nprint("STAGE correctness", flush=True)\ntime.sleep(10)\n', timeout=0.05)
        self.assertEqual(row['status'], 'timeout')
        self.assertLess(row['exit_code'], 0)

    def test_real_signal_is_not_correctness_pass(self):
        row = self.run_fixture('import os,signal\nos.kill(os.getpid(), signal.SIGKILL)\n')
        self.assertEqual(row['status'], 'worker_error')
        self.assertEqual(row['exit_code'], -9)


if __name__ == '__main__':
    unittest.main()
