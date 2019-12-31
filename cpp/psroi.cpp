#include<torch/torch.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include "utils.h"
#include<stdio.h>
namespace py = pybind11;

at::Tensor PSROI_Forward(at::Tensor& fsam,at::Tensor& box,int alpha,int p)
{
    assert(fsam.dim()==4 && box.dim()==3);
    assert(fsam.dtype()==torch::kF32);
    assert(box.dtype()==torch::kInt32);
    assert(fsam.size(1)==alpha*p*p);
    assert(fsam.size(0)==box.size(0)&& box.size(2)==4);
    int batch_size=fsam.size(0);
    int bnd_per_batch=box.size(1);
    at::Device type=fsam.device();
    at::Tensor fsam_cpu=fsam.to(at::kCPU);
    at::Tensor feature=fsam_cpu.contiguous();
    long table_size=fsam.size(0)*fsam.size(1)*(fsam.size(2)+1)*(fsam.size(3)+1);
    float* table=(float*)malloc(sizeof(float)*table_size);
    memset(table,0,sizeof(float)*table_size);
    float* data=feature.data_ptr<float>();
    at::Tensor box_cpu=box.to(at::kCPU);
    int* ptr_box=box_cpu.contiguous().data_ptr<int>();
    //计算积分图
    for(int c=0;c<fsam.size(0)*fsam.size(1);c++)
    {
        float* base=data+c*fsam.size(2)*fsam.size(3);
        float* table_base=table+c*(fsam.size(2)+1)*(fsam.size(3)+1);
        for(int row=1;row<=fsam.size(2);row++)
        {
            float temp=0;
            int inneroffset=(row-1)*fsam.size(3);
            int table_inneroffset=row*(fsam.size(3)+1);
            for(int col=1;col<=fsam.size(3);col++)
            {
                temp+=base[inneroffset+col-1];
                table_base[table_inneroffset+col]=table_base[table_inneroffset-fsam.size(3)-1+col]+temp;
            }
        }
    }
    // for(int c=0;c<fsam.size(0)*fsam.size(1);c++)
    // {
    //     float* table_base=table+c*(fsam.size(2)+1)*(fsam.size(3)+1);
    //     for(int i=0;i<=fsam.size(2);i++)
    //     {
    //         int table_inneroffset=i*(fsam.size(3)+1);
    //         for(int j=0;j<=fsam.size(3);j++)
    //         {
    //             printf("%f ",table_base[table_inneroffset+j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    float* out=psroi_forward_gpu(table,batch_size,fsam.size(1),fsam.size(2)+1,fsam.size(3)+1,ptr_box,bnd_per_batch,p);
    at::Tensor out_tensor=torch::from_blob(out,{batch_size,bnd_per_batch,alpha,p,p},torch::dtype(torch::kFloat32));
    free(table);
    at::Tensor ret=out_tensor.to(type,false,true);
    free(out);
    return ret;
}

at::Tensor PSROI_Backward(at::Tensor& grad_output,at::Tensor& box,int alpha,int p,int w,int h)
{
    assert(grad_output.dim()==2 && box.dim()==3);
    assert(grad_output.dtype()==torch::kF32 && box.dtype()==torch::kInt32);
    int batch_size=box.size(0);
    int bnd_per_batch=box.size(1);
    int channel=alpha*p*p;
    at::Device type=grad_output.device();
    at::Tensor grad_cpu=grad_output.to(at::kCPU);
    at::Tensor grad=grad_cpu.view({batch_size,bnd_per_batch,alpha,p,p}).contiguous();
    at::Tensor box_cpu=box.to(at::kCPU);
    int* ptr_box=box_cpu.contiguous().data_ptr<int>();
    float* ptr_grad=grad.data_ptr<float>();
    float* out=psroi_backward_gpu(ptr_grad,batch_size,channel,w,h,ptr_box,bnd_per_batch,p);
    at::Tensor out_tensor=torch::from_blob(out,{batch_size,channel,w,h},torch::dtype(torch::kF32));
    at::Tensor ret=out_tensor.to(type,false,true);
    free(out);
    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
    m.def("PSROI_Forward",&PSROI_Forward,"compute PSROI forward");
    m.def("PSROI_Backward",&PSROI_Backward,"compute PSROI backward");
}