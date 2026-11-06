"""CPU-only checks of the actual helper, failure paths, patch and audit parser.

CUDA and NUMA syscalls are explicit test doubles here. These tests do NOT
establish GH200 compatibility; the separate GPU runner is the required gate.
"""
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'thesis/scripts'))
from run_torch_host_malloc import validate_log
from torch_host_malloc_common import PACKAGE,PATCH,clean_environment,runtime_environment

CUDA_STUB=r'''
#pragma once
#include <cstring>
typedef int cudaError_t;
typedef void* cudaStream_t;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };
enum {cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaDevAttrPageableMemoryAccessUsesHostPageTables=1,
      cudaDevAttrPageableMemoryAccess=2,cudaDevAttrConcurrentManagedAccess=3};
struct cudaDeviceProp {int major=9;int minor=0;char name[256]="NVIDIA GH200 480GB";};
inline int syncs=0,fail_sync=0,capturing=0;
inline const char* cudaGetErrorString(int){return "mock CUDA status";}
inline int cudaGetDeviceCount(int* p){*p=1;return 0;}
inline int cudaGetDevice(int* p){*p=0;return 0;}
inline int cudaGetDeviceProperties(cudaDeviceProp* p,int){*p=cudaDeviceProp{};return 0;}
inline int cudaDeviceGetAttribute(int* p,int,int){*p=1;return 0;}
inline int cudaStreamIsCapturing(void*,cudaStreamCaptureStatus* p){*p=capturing?cudaStreamCaptureStatusActive:cudaStreamCaptureStatusNone;return 0;}
inline int cudaDeviceSynchronize(){++syncs;return fail_sync;}
inline int cudaStreamSynchronize(void*){++syncs;return fail_sync;}
'''
HARNESS=r'''
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#include <cstdarg>
#include <cassert>
#include <cstdlib>
#include <iostream>
int bad_node=0,resets=0;
long mock_syscall(long number,...) {
  va_list args;va_start(args,number);
  if(number==SYS_mbind) {
    (void)va_arg(args,void*);(void)va_arg(args,size_t);
    int mode=va_arg(args,int);if(mode==MPOL_DEFAULT)++resets;
  } else if(number==SYS_get_mempolicy) {
    *va_arg(args,int*)=MPOL_BIND;*va_arg(args,unsigned long*)=1UL;
  } else if(number==SYS_move_pages) {
    (void)va_arg(args,int);auto n=va_arg(args,size_t);(void)va_arg(args,void**);
    (void)va_arg(args,void*);int* status=va_arg(args,int*);
    for(size_t i=0;i<n;++i)status[i]=bad_node;
  } else {std::abort();}
  va_end(args);return 0;
}
#define syscall mock_syscall
#include "ThesisHostMalloc.h"
#undef syscall
int main(int argc,char**argv) {
  using namespace thesis_host_malloc;
  std::string test=argv[1];
  (void)frozen_mode();
  if(test=="off" || test=="off_unset") {setenv("THESIS_HOST_MALLOC","1",1);assert(!enabled());guard(nullptr);return 0;}
  if(test=="badflag") {try{guard(nullptr);}catch(const std::exception&){return 0;}return 10;}
  assert(enabled());setenv("THESIS_HOST_MALLOC","0",1);assert(enabled());
  guard(nullptr);
  void* p=nullptr;
  if(test=="placement_failure") {
    bad_node=1;try{allocate(&p,100);}catch(const std::exception&){assert(resets==1);assert(state().live==0);return 0;}return 11;
  }
  assert(allocate(&p,12345)==cudaSuccess && p);
  assert(reinterpret_cast<uintptr_t>(p)%state().page==0);
  {std::lock_guard<std::mutex> lock(state().mutex);assert(contains_locked(state(),static_cast<char*>(p)+10,123));}
  void* q=nullptr;assert(allocate(&q,12345)==cudaSuccess);
  std::memset(p,7,12345);copy(q,p,12345,nullptr);assert(std::memcmp(p,q,12345)==0);
  fail_sync=1;
  try{release(p);return 12;}catch(const std::exception&){}
  assert(state().blocks.size()==2);fail_sync=0;
  bad_node=1;try{release(p);return 13;}catch(const std::exception&){}
  assert(state().blocks.size()==2);bad_node=0;
  release(p);release(q);assert(state().live==0 && state().blocks.empty() && syncs>=4);
  assert(resets==2);
  void* z=reinterpret_cast<void*>(1);
  assert(allocate(&z,SIZE_MAX)==cudaErrorMemoryAllocation && !z);
  assert(allocate(&z,state().limit)==cudaErrorMemoryAllocation && !z);
  assert(allocate(&z,0)==cudaSuccess && !z);
  capturing=1;try{guard(nullptr);return 14;}catch(const std::exception&){}
  return 0;
}
'''


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.tmp.name)
        (cls.folder/'cuda_runtime_api.h').write_text(CUDA_STUB)
        (cls.folder/'test.cpp').write_text(HARNESS)
        cls.exe=cls.folder/'test'
        sp.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter',
                '-pthread','-I'+str(cls.folder),'-I'+str(PACKAGE),str(cls.folder/'test.cpp'),'-o',str(cls.exe)],check=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def invoke(self,name,flag):
        log=self.folder/(name+'.jsonl')
        env=dict(os.environ,THESIS_HOST_MALLOC_LOG=str(log),
                 THESIS_HOST_MALLOC_MAX_BYTES=str(1<<20))
        if flag is None:env.pop('THESIS_HOST_MALLOC',None)
        else:env['THESIS_HOST_MALLOC']=flag
        sp.run([self.exe,name],env=env,check=True)
        return log

    def test_default_off_is_inert_and_frozen(self):
        self.assertFalse(self.invoke('off','0').exists())
        self.assertFalse(self.invoke('off_unset',None).exists())

    def test_host_lifetime_alignment_budget_copy_and_failures(self):
        log=self.invoke('host','1')
        result=validate_log(log,1<<20)
        self.assertEqual(result['allocations'],2)
        self.assertEqual(result['frees'],2)
        self.assertEqual(result['retained_bytes'],0)

    def test_failed_placement_does_not_expose_pointer(self):
        log=self.invoke('placement_failure','1')
        self.assertEqual(validate_log(log,1<<20)['allocations'],0)

    def test_invalid_flag_fails_without_log(self):
        self.assertFalse(self.invoke('badflag','yes').exists())

    def test_audit_rejects_inconsistent_or_duplicate_free(self):
        log=self.folder/'badlog.jsonl'
        rows=[dict(event='config',limit=99,node=0),dict(event='free',ptr=123,node=0,live=0)]
        log.write_text('\n'.join(json.dumps(r) for r in rows))
        with self.assertRaises(AssertionError):validate_log(log,99)

    def test_patch_header_is_exactly_the_reviewed_helper(self):
        # Check the /dev/null diff using git itself, not a mirrored parser.
        text=PATCH.read_text()
        header_diff=text[text.index('--- /dev/null\n'):]
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);(folder/'header.patch').write_text(header_diff)
            sp.run(['git','apply','--check','header.patch'],cwd=folder,check=True)
            sp.run(['git','apply','header.patch'],cwd=folder,check=True)
            self.assertEqual((folder/'c10/cuda/ThesisHostMalloc.h').read_bytes(),(PACKAGE/'ThesisHostMalloc.h').read_bytes())

    def test_child_environment_cannot_inherit_pip_target_or_host_mode(self):
        inherited=dict(THESIS_HOST_MALLOC='1',PIP_TARGET='/original/fa3',PYTHONPATH='/wrong',
                       PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',LD_LIBRARY_PATH='/wrong/torch/lib')
        with patch.dict(os.environ,inherited):
            child=clean_environment()
            for key in ['THESIS_HOST_MALLOC','PIP_TARGET','PYTHONPATH','PYTORCH_CUDA_ALLOC_CONF']:
                self.assertNotIn(key,child)
            self.assertEqual(child['PIP_CONFIG_FILE'],os.devnull)
            self.assertNotIn('/wrong/torch/lib',runtime_environment(self.folder)['LD_LIBRARY_PATH'])
            self.assertEqual(os.environ['THESIS_HOST_MALLOC'],'1')
            self.assertEqual(os.environ['PIP_TARGET'],'/original/fa3')

    def test_plan_commands_are_read_only_and_enumerate_all_checks(self):
        target=self.folder/'must_not_be_created'
        for script in ['build_torch_host_malloc.py','run_torch_host_malloc.py']:
            result=sp.run([sys.executable,str(ROOT/'thesis/scripts'/script),'--plan','--root',str(target)],
                          check=True,capture_output=True,text=True)
            self.assertFalse(target.exists())
            if script.startswith('run_'):
                self.assertEqual(len([x for x in result.stdout.splitlines() if 'feature=' in x]),16)


if __name__=='__main__':unittest.main()
