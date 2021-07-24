#include "../ps_roi_align.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

class PSROIAlignFunction
    : public torch::autograd::Function<PSROIAlignFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["input_shape"] = input.sizes();
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = ps_roi_align(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);

    auto output = std::get<0>(result);
    auto channel_mapping = std::get<1>(result);
    ctx->save_for_backward({rois, channel_mapping});
    ctx->mark_non_differentiable({channel_mapping});

    return {output, channel_mapping};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto channel_mapping = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = detail::_ps_roi_align_backward(
        grad_output[0],
        rois,
        channel_mapping,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        ctx->saved_data["sampling_ratio"].toInt(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3]);

    return {
        grad_in,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable()};
  }
};

std::tuple<at::Tensor, at::Tensor> ps_roi_align_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
  auto result = PSROIAlignFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);

  return std::make_tuple(result[0], result[1]);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_align"),
      TORCH_FN(ps_roi_align_autograd));
}

} // namespace ops
} // namespace vision
