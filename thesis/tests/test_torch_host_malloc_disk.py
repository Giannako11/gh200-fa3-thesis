"""Real process-group watchdog tests, without consuming disk space."""
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from torch_host_malloc_disk import DiskSpaceError,run_guarded


def alive(pid):
    # Consult the process namespace first: some executors virtualize child PIDs
    # while exposing a host /proc mount, whose same-number entries are unrelated.
    try:os.getpgid(pid)
    except ProcessLookupError:return False
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


class WatchdogTests(unittest.TestCase):
    def test_low_disk_does_not_launch_command(self):
        with patch('torch_host_malloc_disk.shutil.disk_usage',return_value=types.SimpleNamespace(free=0)), patch('torch_host_malloc_disk.sp.Popen') as launch:
            with self.assertRaises(DiskSpaceError):
                run_guarded(['unused'],cwd='.',env=os.environ,log=None,root='.',reserve_bytes=1)
            launch.assert_not_called()

    def test_success_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as t, patch('torch_host_malloc_disk.shutil.disk_usage',return_value=types.SimpleNamespace(free=100)):
            with (Path(t)/'log').open('w') as log:
                run_guarded([sys.executable,'-c','print("done")'],cwd=t,env=os.environ,log=log,root=t,reserve_bytes=1,poll_seconds=0.01)
                with self.assertRaises(sp.CalledProcessError) as error:
                    run_guarded([sys.executable,'-c','raise SystemExit(9)'],cwd=t,env=os.environ,log=log,root=t,reserve_bytes=1,poll_seconds=0.01)
                self.assertEqual(error.exception.returncode,9)
            self.assertIn('done',(Path(t)/'log').read_text())

    def test_build_tree_stopped_and_unrelated_process_left_running(self):
        sentinel=sp.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as t:
                marker=Path(t)/'pids.json'
                script=('import subprocess,sys,os,json,time,pathlib; '
                        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                        'pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(30)')
                def space(_):return types.SimpleNamespace(free=0 if marker.exists() else 100)
                with patch('torch_host_malloc_disk.shutil.disk_usage',side_effect=space):
                    with (Path(t)/'log').open('w') as log:
                        with self.assertRaises(DiskSpaceError):
                            run_guarded([sys.executable,'-c',script,str(marker)],cwd=t,env=os.environ,log=log,root=t,reserve_bytes=1,poll_seconds=0.02)
                pids=json.loads(marker.read_text())
                deadline=time.monotonic()+1
                while any(alive(pid) for pid in pids) and time.monotonic()<deadline:
                    time.sleep(0.01)
                for pid in pids:self.assertFalse(alive(pid))
                self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate();sentinel.wait(timeout=5)

    def test_plan_defaults_and_invalid_limits(self):
        script=SCRIPTS/'build_torch_host_malloc.py'
        result=sp.run([sys.executable,str(script),'--plan'],check=True,capture_output=True,text=True)
        self.assertIn('>= 30 GiB',result.stdout)
        self.assertIn('below 8 GiB',result.stdout)
        result=sp.run([sys.executable,str(script),'--plan','--min-free-gib','8'],capture_output=True)
        self.assertEqual(result.returncode,2)


if __name__=='__main__':unittest.main()
