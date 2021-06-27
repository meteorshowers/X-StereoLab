// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void BuildCostVolumeForward(const int nthreads, 
    const T* left, const T* right, const T* shift, 
    const int num_batch, const int channels, const int height,
    const int width, const int max_disp,
    T* cost) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int pd = (index / width / height) % max_disp;
    int c = (index / width / height/ max_disp) % channels;
    int n = index / width / height / max_disp / channels;

    int index_L = (((n * 2 * channels + c) * max_disp + pd) * height + ph) * width + pw;
    int index_R = index_L + channels * max_disp * height * width;

    T shift_pd = -shift[n * max_disp + pd];

    cost[index_L] = left[((n * channels + c) * height + ph) * width + pw];

    if (pw + shift_pd >= 0. && pw + shift_pd <= width - 1)
    {
        const T* offset_right = right + (n * channels + c) * height * width;
        cost[index_R] = bilinear_interpolate(offset_right, height, width, (T)ph, (T)pw + shift_pd);
    }
    else 
    {
        cost[index_R] = 0.;
    }
  }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void BuildCostVolumeBackwardFeature(const int nthreads, 
    const T* grad, const T* shift, 
    const int num_batch, const int channels, const int height,
    const int width, const int max_disp,
    T* grad_left, T* grad_right) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int pd = (index / width / height) % max_disp;
    int c = (index / width / height/ max_disp) % channels;
    int n = index / width / height / max_disp / channels;

    int index_L = (((n * 2 * channels + c) * max_disp + pd) * height + ph) * width + pw;
    int index_R = index_L + channels * max_disp * height * width;

    T shift_pd = -shift[n * max_disp + pd];

    // left
    atomicAdd(grad_left + ((n * channels + c) * height + ph) * width + pw, static_cast<T>(grad[index_L]));

    if (pw + shift_pd >= 0. && pw + shift_pd <= width - 1)
    {
        // right
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, (T) ph, (T) pw + shift_pd,
            w1, w2, w3, w4,
            x_low, x_high, y_low, y_high);

        T top_diff_this_bin = grad[index_R];
        T g1 = top_diff_this_bin * w1;
        T g2 = top_diff_this_bin * w2;
        T g3 = top_diff_this_bin * w3;
        T g4 = top_diff_this_bin * w4;

        T* offset_grad_right = grad_right + (n * channels + c) * height * width;
        if (w1 >= 1e-10)
            atomicAdd(offset_grad_right + y_low * width + x_low, static_cast<T>(g1));
        if (w2 >= 1e-10)
            atomicAdd(offset_grad_right + y_low * width + x_high, static_cast<T>(g2));
        if (w3 >= 1e-10)
            atomicAdd(offset_grad_right + y_high * width + x_low, static_cast<T>(g3));
        if (w4 >= 1e-10)
            atomicAdd(offset_grad_right + y_high * width + x_high, static_cast<T>(g4));
    }
  } // CUDA_1D_KERNEL_LOOP
} // BuildCostVolumeBackward


at::Tensor BuildCostVolume_forward_cuda(const at::Tensor& left,
                                 const at::Tensor& right,
                                 const at::Tensor& shift) {
  AT_ASSERTM(left.type().is_cuda(), "left must be a CUDA tensor");
  AT_ASSERTM(right.type().is_cuda(), "right must be a CUDA tensor");
  AT_ASSERTM(shift.type().is_cuda(), "shift must be a CUDA tensor");

  AT_ASSERTM((left.size(0) == right.size(0)) && (left.size(1) == right.size(1)) && \
    (left.size(2) == right.size(2)) && (left.size(3) == right.size(3)), \
    "Left image and right image should match their size.");
  AT_ASSERTM(left.size(0) == shift.size(0), \
    "Image and shift should of same batch.");

  auto num_batch = left.size(0);
  auto channels = left.size(1);
  auto height = left.size(2);
  auto width = left.size(3);
  auto max_disp = shift.size(1);

  auto output = at::empty({num_batch, channels * 2, max_disp, height, width}, left.options());
  auto output_size = num_batch * channels * 2 * max_disp * height * width;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)(output_size / 2), 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(left.type(), "BuildCostVolume_forward", [&] {
    BuildCostVolumeForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size / 2,
         left.contiguous().data<scalar_t>(),
         right.contiguous().data<scalar_t>(),
         shift.contiguous().data<scalar_t>(),
         num_batch,
         channels,
         height,
         width,
         max_disp,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
std::tuple<at::Tensor, at::Tensor> BuildCostVolume_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& shift) {
  AT_ASSERTM(shift.type().is_cuda(), "shift must be a CUDA tensor");

  auto num_batch = grad.size(0);
  auto channels = grad.size(1) / 2;
  auto height = grad.size(3);
  auto width = grad.size(4);
  auto max_disp = shift.size(1);

  auto grad_left = at::zeros({num_batch, channels, height, width}, grad.options());
  auto grad_right = at::zeros({num_batch, channels, height, width}, grad.options());

  AT_ASSERTM(grad.numel() == num_batch * channels * 2 * max_disp * height * width,
      "grad shape is wrong");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(grad_left, grad_right);
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "BuildCostVolume_backward", [&] {
    BuildCostVolumeBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel() / 2,
         grad.contiguous().data<scalar_t>(),
         shift.contiguous().data<scalar_t>(),
         num_batch,
         channels,
         height,
         width,
         max_disp,
         grad_left.data<scalar_t>(),
         grad_right.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(grad_left, grad_right);
}

