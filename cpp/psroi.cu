#include<cuda_runtime.h>
#include<stdio.h>

#define THREADS_PER_BLOCK 256
#define CUDA_GET_BLOCK(n) ((n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK)
#define CUDA_KERNEL_LOOP(i,n) \
    for (int i=blockIdx.x * blockDim.x+threadIdx.x;i<(n);i+=blockDim.x*gridDim.x)

void check_error(cudaError_t state)
{
    if(state!=cudaSuccess)
    {
        const char* s=cudaGetErrorString(state);
        fprintf(stderr,"CUDA ERROR:%s",s);
        exit(0);
    }
}

__global__ void psroi_forward_gpu_kernel(long total,float* out,float* table_gpu,int* bnd_gpu,int batchSize,int channels,int w,int h,int bndPerbatch,int p)
{
    CUDA_KERNEL_LOOP(index,total)
    {
        int alpha=channels/(p*p);
        int batch=index/(bndPerbatch*channels);
        int offset=index%(bndPerbatch*channels);
        int boxth=offset/channels;
        offset=offset%channels;
        int k=offset/(p*p);
        offset=offset%(p*p);
        int y=offset/p;
        int x=offset%p;
        int c=(y*p+x)*alpha+k; //对应积分表的通道
        int* box=bnd_gpu+batch*bndPerbatch*4+boxth*4;
        float* base=table_gpu+batch*channels*w*h+c*w*h;
        int ltx=x*box[2]/p+box[0];
        int lty=y*box[3]/p+box[1];
        int rdx=(x+1)*box[2]/p+box[0];
        int rdy=(y+1)*box[3]/p+box[1];
        if(rdx==ltx)rdx=rdx+1;
        if(rdy==lty)rdy=rdy+1;
        int num=(rdx-ltx)*(rdy-lty);
        out[index]=(base[rdx+rdy*w]+base[ltx+lty*w]-base[rdx+lty*w]-base[ltx+rdy*w])/num;
    }

}

float* psroi_forward_gpu(float* table,int batchSize,int channels,int w,int h,int* bnd,int bndPerbatch,int p)
{
    if(!table || !bnd)return NULL;
    float* table_gpu=NULL,*out=NULL,*out_gpu=NULL;
    int *bnd_gpu=NULL;
    long count1=batchSize*channels*w*h;
    check_error(cudaMalloc(&table_gpu,sizeof(float)*count1));
    check_error(cudaMemcpy(table_gpu,table,count1*sizeof(float),cudaMemcpyHostToDevice));
    long count2=batchSize*bndPerbatch*4;
    check_error(cudaMalloc(&bnd_gpu,sizeof(int)*count2));
    check_error(cudaMemcpy(bnd_gpu,bnd,count2*sizeof(int),cudaMemcpyHostToDevice));
    long count3=bndPerbatch*batchSize*channels;
    check_error(cudaMalloc(&out_gpu,sizeof(float)*count3));
    check_error(cudaMemset(out_gpu,0,sizeof(float)*count3));
    psroi_forward_gpu_kernel<<<CUDA_GET_BLOCK(count3),THREADS_PER_BLOCK>>>(count3,out_gpu,table_gpu,bnd_gpu,batchSize,channels,w,h,bndPerbatch,p);
    check_error(cudaGetLastError());
    out=(float*)malloc(sizeof(float)*count3);
    check_error(cudaMemcpy(out,out_gpu,sizeof(float)*count3,cudaMemcpyDeviceToHost));
    cudaFree(table_gpu);
    cudaFree(bnd_gpu);
    cudaFree(out_gpu);
    return out;
}

__global__ void psroi_backward_gpu_kernel(long total,float* out,int channels,int w,int h,float* grad_gpu,int bndPerbatch,int p,int* bnd_gpu)
{
    CUDA_KERNEL_LOOP(index,total)
    {
        int index=blockDim.x*blockIdx.x+threadIdx.x;
        int alpha=channels/(p*p);
        int batch=index/(bndPerbatch*channels);
        int offset=index%(bndPerbatch*channels);
        int boxth=offset/channels;
        offset=offset%channels;
        int k=offset/(p*p);
        offset=offset%(p*p);
        int y=offset/p;
        int x=offset%p;
        int c=(y*p+x)*alpha+k; 
        int* box=bnd_gpu+batch*bndPerbatch*4+boxth*4;
        float* base=out+batch*channels*w*h+c*w*h;
        int ltx=x*box[2]/p+box[0];
        int lty=y*box[3]/p+box[1];
        int rdx=(x+1)*box[2]/p+box[0]-1;
        int rdy=(y+1)*box[3]/p+box[1]-1;
        if(ltx>rdx)rdx=rdx+1;
        if(lty>rdy)rdy=rdy+1;
        int num=(rdx-ltx+1)*(rdy-lty+1);
        float temp;
        for(int i=lty;i<=rdy;i++)
        {
            for(int j=ltx;j<=rdx;j++)
            {
                temp=grad_gpu[index]/num;
                atomicAdd(base+i*w+j,temp);
            }
        }
    }
}

float* psroi_backward_gpu(float* grad,int batchSize,int channels,int w,int h,int* bnd,int bndPerbatch,int p)
{
    if(!grad || !bnd)return NULL;
    float* grad_gpu=NULL,*out=NULL,*out_gpu=NULL;
    int *bnd_gpu=NULL;
    long count1=batchSize*bndPerbatch*channels;
    check_error(cudaMalloc(&grad_gpu,sizeof(float)*count1));
    check_error(cudaMemcpy(grad_gpu,grad,count1*sizeof(float),cudaMemcpyHostToDevice));
    long count2=batchSize*bndPerbatch*4;
    check_error(cudaMalloc(&bnd_gpu,sizeof(int)*count2));
    check_error(cudaMemcpy(bnd_gpu,bnd,count2*sizeof(int),cudaMemcpyHostToDevice));
    long count3=batchSize*channels*w*h;
    check_error(cudaMalloc(&out_gpu,sizeof(float)*count3));
    check_error(cudaMemset(out_gpu,0,sizeof(float)*count3));
    psroi_backward_gpu_kernel<<<CUDA_GET_BLOCK(count1),THREADS_PER_BLOCK>>>(count1,out_gpu,channels,w,h,grad_gpu,bndPerbatch,p,bnd_gpu);
    out=(float*)malloc(sizeof(float)*count3);
    check_error(cudaMemcpy(out,out_gpu,sizeof(float)*count3,cudaMemcpyDeviceToHost));
    cudaFree(grad_gpu);
    cudaFree(out_gpu);
    cudaFree(bnd_gpu);
    return out;
}