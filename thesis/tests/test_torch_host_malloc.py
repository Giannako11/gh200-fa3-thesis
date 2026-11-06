#!/usr/bin/env python3
"""GH200 subprocess worker. CUDA tests are invoked by run_torch_host_malloc.py."""
import argparse
import ctypes as C
import gc
import json
import os
from pathlib import Path
import sys
import time


def events():
    path = os.environ.get('THESIS_HOST_MALLOC_LOG')
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if path and Path(path).is_file() else []


def allocations():
    alive = {}
    for row in events():
        if row['event'] == 'alloc':
            alive[row['ptr']] = row
        elif row['event'] == 'free':
            assert row['ptr'] in alive, 'Unknown/duplicate release in allocation log'
            alive.pop(row['ptr'])
    return alive


def placement(tensor):
    ptr = tensor.untyped_storage().data_ptr()
    size = tensor.untyped_storage().nbytes()
    if not size:
        return dict(bytes=0, pages=0)
    segment = next((v for k,v in allocations().items()
                    if k <= ptr and ptr+size <= k+v['mapped_bytes']), None)
    assert segment, 'Tensor storage not covered by a live malloc segment'
    page = os.sysconf('SC_PAGE_SIZE')
    start = ptr // page * page
    end = (ptr+size+page-1) // page * page
    lib = C.CDLL('libnuma.so.1', use_errno=True)
    query = lib.numa_move_pages
    query.argtypes = [C.c_int, C.c_ulong, C.POINTER(C.c_void_p), C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int]
    query.restype = C.c_long
    count = 0
    for offset in range(start, end, 512*page):
        addresses = list(range(offset, min(end,offset+512*page), page))
        pages = (C.c_void_p*len(addresses))(*addresses)
        status = (C.c_int*len(addresses))(*([-9999]*len(addresses)))
        rc = query(0, len(addresses), pages, None, status, 0)
        assert rc == 0, f'move_pages failed: rc={rc} errno={C.get_errno()}'
        assert set(status) == {0}, f'CPU placement failure: {set(status)}'
        count += len(addresses)
    return dict(ptr=ptr, bytes=size, pages=count, node=0, segment_ptr=segment['ptr'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--host', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    import torch
    result=dict(case=args.case, host=args.host, torch_version=torch.__version__,
                torch_commit=torch.version.git_version, python=sys.executable,
                checks=[], placement=[], elapsed_seconds=None)
    started=time.monotonic()
    expected_errors={'bad_flag':'flag must be exactly', 'backend_async':'backend must be native',
                     'expandable':'expandable_segments must be false'}
    if args.case in expected_errors:
        try:
            torch.empty(1,device='cuda')
        except RuntimeError as e:
            assert expected_errors[args.case] in str(e),str(e)
            result['checks'].append('unsupported configuration rejected')
        else:
            raise AssertionError('Unsupported configuration silently accepted')
    else:
        if args.case=='frozen_on':
            os.environ['THESIS_HOST_MALLOC']='0'
        assert torch.cuda.is_available() and 'GH200' in torch.cuda.get_device_name(0)
        assert torch.cuda.device_count()==1
        assert torch.cuda.get_allocator_backend()=='native'
        torch.backends.cuda.matmul.allow_tf32=False
        if args.case=='frozen_off':
            x=torch.empty(1,device='cuda')
            os.environ['THESIS_HOST_MALLOC']='1'
            y=torch.ones(17,device='cuda'); torch.cuda.synchronize()
            assert not events(),'Late environment mutation activated backend'
            result['checks'].append('OFF remains OFF after environment mutation')
        elif args.case=='oom':
            x=torch.ones(1024,device='cuda'); torch.cuda.synchronize()
            try:
                torch.empty(1<<30,device='cuda',dtype=torch.uint8)
            except (torch.OutOfMemoryError,RuntimeError) as e:
                assert 'memory' in str(e).lower(),str(e)
                assert any(x['event']=='budget_reject' for x in events())
            else:
                raise AssertionError('Host budget not enforced')
            x.add_(2); torch.cuda.synchronize()
            assert float(x.sum().item())==3072
            result['checks'].append('budget OOM is recoverable; no huge host allocation touched')
        elif args.case=='graph_reject':
            x=torch.ones(32,device='cuda'); torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g):
                    y=x+1
            except RuntimeError as e:
                assert 'THESIS_HOST_MALLOC' in str(e),str(e)
            else:
                raise AssertionError('CUDA graph accepted unexpectedly')
            result['checks'].append('graph rejected')
        elif args.case=='reconfigure_reject':
            x=torch.ones(32,device='cuda'); torch.cuda.synchronize()
            torch.cuda.memory._set_allocator_settings('backend:native,expandable_segments:True')
            try:
                torch.empty(1,device='cuda')
            except RuntimeError as e:
                assert 'expandable_segments must be false' in str(e),str(e)
            else:
                raise AssertionError('Runtime reconfiguration bypassed host backend')
            finally:
                torch.cuda.memory._set_allocator_settings('backend:native,expandable_segments:False')
            result['checks'].append('runtime reconfiguration cannot bypass host path, even for cached blocks')
        elif args.case=='fraction_reject':
            torch.empty(1,device='cuda')
            try:
                torch.cuda.set_per_process_memory_fraction(0.5)
            except RuntimeError as e:
                assert 'THESIS_HOST_MALLOC' in str(e),str(e)
            else:
                raise AssertionError('HBM fraction accepted in host mode')
            result['checks'].append('HBM-based fraction rejected')
        else:
            # Populate context/library workspaces before physical-memory observation.
            a=torch.ones((16,16),device='cuda'); b=a@a
            assert float(b.sum().item())==4096
            del a,b
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            if args.case in ('standard','frozen_on','uncached'):
                for dtype in (torch.float32,torch.float16,torch.bfloat16,torch.int64):
                    for n in (0,1,17,1024,100003):
                        x=torch.empty(n,device='cuda',dtype=dtype)
                        if args.host:
                            result['placement'].append(dict(phase='before',**placement(x)))
                        x.fill_(2); x.add_(3); torch.cuda.synchronize()
                        assert torch.equal(x.cpu(),torch.full((n,),5,dtype=dtype))
                        if args.host:
                            result['placement'].append(dict(phase='after',**placement(x)))
                        del x
                result['checks'].append('CUDA creation/fill/add for four dtypes and zero/odd/large sizes')
                x=torch.zeros(65536,device='cuda',dtype=torch.float32)
                torch.cuda.synchronize()
                if args.host:
                    result['placement'].append(dict(phase='direct_before',**placement(x)))
                    view=(C.c_float*x.numel()).from_address(x.data_ptr())
                    for i in range(x.numel()): view[i]=float(i%7)
                    # CUDA dispatch executes the ordinary ATen elementwise kernel.
                    x.mul_(2).add_(1); torch.cuda.synchronize()
                    assert all(view[i]==2*(i%7)+1 for i in range(x.numel()))
                    result['placement'].append(dict(phase='direct_after',**placement(x)))
                    del view
                    result['checks'].append('CPU writes -> GPU kernel -> CPU reads at identical data_ptr')
                cpu=torch.arange(4096,dtype=torch.float32).reshape(64,64)
                x=cpu.to('cuda'); clone=x.clone(); clone.copy_(x)
                assert torch.equal(clone.cpu(),cpu)
                transpose=x.t().contiguous()
                assert torch.equal(transpose.cpu(),cpu.t().contiguous())
                assert float(x.sum().item())==float(cpu.sum())
                assert x[:4].untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
                if args.host:
                    for value in [x,clone,transpose]: result['placement'].append(placement(value))
                del x,clone,transpose
                for dtype in (torch.float32,torch.float16,torch.bfloat16):
                    ac=(torch.arange(128*128).reshape(128,128)%7-3).to(dtype)/8
                    bc=(torch.arange(128*128).reshape(128,128)%5-2).to(dtype)/8
                    ag,bg=ac.to('cuda'),bc.to('cuda')
                    out=ag@bg
                    torch.testing.assert_close(out.cpu().float(),ac.float()@bc.float(),rtol=0.01,atol=0.02)
                    if args.host:
                        for value in [ag,bg,out]: result['placement'].append(placement(value))
                    del ag,bg,out,ac,bc
                result['checks'].append('copies/views/reduction and FP32/FP16/BF16 cuBLAS matmul')
                gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
                before=torch.cuda.mem_get_info()[0]
                x=torch.empty(8*1024*1024,device='cuda',dtype=torch.float32)
                x.fill_(1); x.add_(1); torch.cuda.synchronize()
                after=torch.cuda.mem_get_info()[0]
                pointer=x.data_ptr()
                if args.host: result['placement'].append(placement(x))
                result['physical_memory_observation']=dict(tensor_bytes=x.numel()*x.element_size(),
                    free_hbm_before=before,free_hbm_after=after,hbm_delta=before-after,
                    logical_allocated=torch.cuda.memory_allocated(),logical_reserved=torch.cuda.memory_reserved())
                if args.host:
                    assert before-after < 16*1024*1024,'32 MiB host tensor unexpectedly consumed substantial HBM'
                else:
                    assert before-after >= 16*1024*1024,'Stock/OFF 32 MiB tensor did not show expected HBM allocation'
                del x; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
                if args.host:
                    assert not any(k<=pointer<k+v['mapped_bytes'] for k,v in allocations().items()),'Large block retained after empty_cache'
                result['checks'].append('large segment released after GPU completion and empty_cache')
            elif args.case=='streams':
                s=torch.cuda.Stream()
                for i in range(20):
                    x=torch.full((16384,),float(i),device='cuda')
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        torch.cuda._sleep(1000000)
                        y=x*3
                    x.record_stream(s)
                    del x; gc.collect()
                    pressure=torch.empty(32768,device='cuda').fill_(-1)
                    torch.cuda.empty_cache()
                    s.synchronize()
                    assert float(y.sum().item())==float(i*3*16384)
                    if args.host: result['placement'].append(placement(y))
                    del y,pressure
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                result['checks'].append('20 cross-stream record_stream/reuse/release iterations')
            else:
                raise AssertionError('Unknown test case: '+args.case)
        if args.host:
            assert any(x['event']=='alloc' for x in events()),'Host backend not reached'
            result['remaining_segments']=len(allocations())
        else:
            assert not events(),'Feature-OFF execution touched host backend'
    result.update(status='PASS',elapsed_seconds=round(time.monotonic()-started,3))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='placement'},indent=2))


if __name__=='__main__':
    main()
