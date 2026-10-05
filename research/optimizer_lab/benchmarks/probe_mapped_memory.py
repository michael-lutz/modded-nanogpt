"""Small executable probe: can Triton directly access registered Grace host RAM?"""
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+i,tl.load(X+i,i<N,0),i<N)

if __name__=='__main__':
    x=torch.arange(4096,dtype=torch.float32,pin_memory=True)
    y=torch.empty_like(x,device='cuda')
    copy_kernel[(4,)](x,y,4096,1024)
    torch.cuda.synchronize()
    torch.testing.assert_close(x,y.cpu())
    copy_kernel[(4,)](y,x,4096,1024)
    torch.cuda.synchronize()
    print('Mapped pinned host GPU read/write works',flush=True)
