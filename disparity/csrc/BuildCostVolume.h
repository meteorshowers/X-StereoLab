#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor BuildCostVolume_forward(const at::Tensor& left,
                            const at::Tensor& right,
                            const at::Tensor& shift) {
  if (left.type().is_cuda()) {
#ifdef WITH_CUDA
    return BuildCostVolume_forward_cuda(left, right, shift);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> BuildCostVolume_backward(const at::Tensor& grad,
                             const at::Tensor& shift) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return BuildCostVolume_backward_cuda(grad, shift);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

