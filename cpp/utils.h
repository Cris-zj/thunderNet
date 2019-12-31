#ifndef _UTILS_H_
#define _UTILS_H_

float* psroi_forward_gpu(float* table,int batchSize,int channels,int w,int h,int* bnd,int bndPerbatch,int p);

float* psroi_backward_gpu(float* grad,int batchSize,int channels,int w,int h,int* bnd,int bndPerbatch,int p);
#endif