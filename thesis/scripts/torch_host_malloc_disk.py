"""Disk watchdog for commands owned by this build, not other users' processes."""
import os
import shutil
import signal
import subprocess as sp


class DiskSpaceError(RuntimeError):
    pass


def require_reserve(root, reserve_bytes):
    free = shutil.disk_usage(root).free
    if free < reserve_bytes:
        raise DiskSpaceError(
            f'Build stopped: {free/(1<<30):.2f} GiB free on {root}; '
            f'reserve is {reserve_bytes/(1<<30):.2f} GiB. '
            'Build artifacts are retained; free space before retrying.')


def stop_group(process):
    # The command and its compiler/download children have their own session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except sp.TimeoutExpired:
        pass
    # The group leader can exit before one of its descendants. Kill any group
    # members still alive, without touching the caller or unrelated processes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_guarded(argv, *, cwd, env, log, root, reserve_bytes, poll_seconds=0.5):
    require_reserve(root, reserve_bytes)
    process = sp.Popen([str(x) for x in argv], cwd=cwd, env=env,
                       stdout=log, stderr=sp.STDOUT, start_new_session=True)
    try:
        while process.poll() is None:
            require_reserve(root, reserve_bytes)
            try:
                process.wait(timeout=poll_seconds)
            except sp.TimeoutExpired:
                continue
        require_reserve(root, reserve_bytes)
        if process.returncode:
            raise sp.CalledProcessError(process.returncode, argv)
    except BaseException:
        stop_group(process)
        raise
