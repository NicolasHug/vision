import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import grid_sample, make_coords_grid


class ResidualBlock(nn.Module):
    # This is pretty similar to resnet.BasicBlock except for one call to relu, and the bias terms
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super().__init__()

        # Note regarding bias=True:
        # Usually we can pass bias=False in conv layers followed by a norm layer.
        # But in the RAFT training reference, the BatchNorm2d layers are only activated for the first dataset,
        # and frozen for the rest of the training process (i.e. set as eval()). The bias term is thus still useful
        # for the rest of the datasets. Technically, we could remove the bias for other norm layers like Instance norm
        # because these aren't frozen, but we don't bother.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        downsample = stride != 1

        if norm_layer is not None:
            self.norm1 = norm_layer(out_channels)
            self.norm2 = norm_layer(out_channels)
            if downsample:
                self.norm3 = norm_layer(out_channels)
        else:
            self.norm1 = self.norm2 = self.norm3 = nn.Sequential()  # pass-through

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True), self.norm3
            )
        else:
            self.downsample = None

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        downsample = stride != 1
        if norm_layer is not None:
            self.norm1 = norm_layer(out_channels // 4)
            self.norm2 = norm_layer(out_channels // 4)
            self.norm3 = norm_layer(out_channels)
            if downsample:
                self.norm4 = norm_layer(out_channels)
        else:
            self.norm1 = self.norm2 = self.norm3 = self.norm4 = nn.Sequential()  # pass-through

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), self.norm4
            )
        else:
            self.downsample = None

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class FeatureEncoder(nn.Module):
    # TODO: "layers" name is to keep consistent with resnet. Not sure it's the best name
    def __init__(self, block=ResidualBlock, layers=(64, 64, 96, 128, 256), norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert len(layers) == 5

        # see note in ResidualBlock regarding the bias
        self.conv1 = nn.Conv2d(3, layers[0], kernel_size=7, stride=2, padding=3, bias=True)
        self.norm1 = norm_layer(layers[0]) if norm_layer is not None else nn.Sequential()
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_layer=norm_layer, first_stride=1)
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_layer=norm_layer, first_stride=2)
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_layer=norm_layer, first_stride=2)

        self.conv2 = nn.Conv2d(layers[3], layers[4], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_2_blocks(self, block, in_channels, out_channels, norm_layer, first_stride):
        block1 = block(in_channels, out_channels, norm_layer=norm_layer, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        return x


class MotionEncoder(nn.Module):
    def __init__(self, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128):
        super().__init__()

        assert len(flow_layers) == 2
        assert len(corr_layers) in (1, 2)

        self.convc1 = nn.Conv2d(in_channels_corr, corr_layers[0], kernel_size=1, padding=0)
        if len(corr_layers) == 2:
            self.convc2 = nn.Conv2d(corr_layers[0], corr_layers[1], kernel_size=3, padding=1)
        else:
            self.convc2 = None

        self.convf1 = nn.Conv2d(2, flow_layers[0], kernel_size=7, padding=3)
        self.convf2 = nn.Conv2d(flow_layers[0], flow_layers[1], kernel_size=3, padding=1)

        self.conv = nn.Conv2d(
            corr_layers[-1] + flow_layers[-1], out_channels - 2, 3, padding=1
        )  # -2 because we cat the flow

        self.out_channels = out_channels

    def forward(self, flow, corr_features):
        corr = F.relu(self.convc1(corr_features))
        if self.convc2 is not None:
            corr = F.relu(self.convc2(corr))

        flow_orig = flow
        flow = F.relu(self.convf1(flow))
        flow = F.relu(self.convf2(flow))

        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = F.relu(self.conv(corr_flow))
        return torch.cat([corr_flow, flow_orig], dim=1)


class ConvGRU(nn.Module):
    def __init__(self, *, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class ReccurrentBlock(nn.Module):
    def __init__(self, *, input_size, hidden_size, kernel_size=((1, 5), (5, 1)), padding=((0, 2), (2, 0))):
        super().__init__()

        assert len(kernel_size) == len(padding)
        assert len(kernel_size) in (1, 2)

        self.convgru1 = ConvGRU(
            input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[0], padding=padding[0]
        )
        if len(kernel_size) == 2:
            self.convgru2 = ConvGRU(
                input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[1], padding=padding[1]
            )
        else:
            self.convgru2 = None

        self.hidden_size = hidden_size

    def forward(self, h, x):
        h = self.convgru1(h, x)
        if self.convgru2 is not None:
            h = self.convgru2(h, x)
        return h


class FlowHead(nn.Module):
    def __init__(self, *, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, *, motion_encoder, reccurrent_block, flow_head):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.reccurrent_block = reccurrent_block
        self.flow_head = flow_head

        self.hidden_state_size = reccurrent_block.hidden_size

    def forward(self, hidden_state, context, corr_features, flow):
        motion_features = self.motion_encoder(flow, corr_features)
        x = torch.cat([context, motion_features], dim=1)

        hidden_state = self.reccurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return hidden_state, delta_flow


class MaskPredictor(nn.Module):
    def __init__(self, *, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 8 * 8 * 9 because the predicted flow is downsampled by 8, from the downsampling of the initial FeatureEncoder
        # and we interpolate with all 9 surrounding neighbors. See paper and appendix
        self.conv2 = nn.Conv2d(hidden_size, 8 * 8 * 9, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return 0.25 * x


class CorrBlock:
    def __init__(self, *, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        # The neighborhood of a centroid pixel x' is {x' + delta, ||delta||_inf <= radius}
        # so it's a square surrounding x', and its sides have a length of 2 * radius + 1
        # The paper claims that it's ||.||_1 instead of ||.||_inf but the original code uses infinity-norm.
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        def compute_corr_volume(fmap1, fmap2):
            batch_size, num_channels, h, w = fmap1.shape
            fmap1 = fmap1.view(batch_size, num_channels, h * w)
            fmap2 = fmap2.view(batch_size, num_channels, h * w)

            corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
            corr = corr.view(batch_size, h, w, 1, h, w)
            return corr / torch.sqrt(torch.tensor(num_channels))

        torch._assert(fmap1.shape == fmap2.shape, "Input feature maps should have the same shapes")
        corr_volume = compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, num_channels, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), axis=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, neigh_size, neigh_size, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                batch_size, h, w, -1
            )
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        torch._assert(
            corr_features.shape == (batch_size, self.out_channels, h, w),
            "Output shape of index pyramid is incorrect",
        )

        return corr_features


def raft_small():
    features_layers = (32, 32, 64, 96, 128)
    feature_encoder = FeatureEncoder(block=BottleneckBlock, layers=features_layers, norm_layer=nn.InstanceNorm2d)
    context_layers = (32, 32, 64, 96, 160)
    context_encoder = FeatureEncoder(block=BottleneckBlock, layers=context_layers, norm_layer=None)

    num_levels = 4
    radius = 3
    corr_block = CorrBlock(num_levels=num_levels, radius=radius)

    motion_encoder = MotionEncoder(
        in_channels_corr=corr_block.out_channels, corr_layers=(96,), flow_layers=(64, 32), out_channels=82
    )

    hidden_state_size = 96
    out_channels_context = context_layers[-1] - hidden_state_size  # See comments in forward pas of RAFT class
    reccurrent_block = ReccurrentBlock(
        input_size=motion_encoder.out_channels + out_channels_context,
        hidden_size=hidden_state_size,
        kernel_size=(3,),
        padding=(1,),
    )

    flow_head = FlowHead(in_channels=hidden_state_size, hidden_size=128)

    update_block = UpdateBlock(motion_encoder=motion_encoder, reccurrent_block=reccurrent_block, flow_head=flow_head)

    mask_predictor = None

    return RAFT(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor,
    )


def raft():

    features_layers = context_layers = (64, 64, 96, 128, 256)
    feature_encoder = FeatureEncoder(block=ResidualBlock, layers=features_layers, norm_layer=nn.InstanceNorm2d)
    context_encoder = FeatureEncoder(block=ResidualBlock, layers=context_layers, norm_layer=nn.BatchNorm2d)

    num_levels = 4
    radius = 4
    corr_block = CorrBlock(num_levels=num_levels, radius=radius)

    motion_encoder = MotionEncoder(
        in_channels_corr=corr_block.out_channels, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128
    )

    hidden_state_size = 128
    out_channels_context = context_layers[-1] - hidden_state_size  # See comments in forward pas of RAFT class
    reccurrent_block = ReccurrentBlock(
        input_size=motion_encoder.out_channels + out_channels_context,
        hidden_size=hidden_state_size,
        kernel_size=((1, 5), (5, 1)),
        padding=((0, 2), (2, 0)),
    )

    flow_head = FlowHead(in_channels=hidden_state_size, hidden_size=256)

    update_block = UpdateBlock(motion_encoder=motion_encoder, reccurrent_block=reccurrent_block, flow_head=flow_head)

    mask_predictor = MaskPredictor(in_channels=hidden_state_size, hidden_size=256)

    return RAFT(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor,
    )


class RAFT(nn.Module):
    def __init__(self, *, feature_encoder, context_encoder, corr_block, update_block, mask_predictor=None):
        super().__init__()

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block

        self.mask_predictor = mask_predictor

        if not hasattr(self.update_block, "hidden_state_size"):
            raise ValueError("The update_block parameter should expose a 'hidden_state_size' attribute.")

    def _upsample_flow(self, flow, up_mask=None):
        """Upsample flow by a factor of 8, using convex combination weights from up_mask.

        If up_mask is None we just interpolate.
        """
        batch_size, _, h, w = flow.shape
        new_h, new_w = h * 8, w * 8

        if up_mask is None:
            return 8 * F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)

        # See paper page 8 and appendix B.
        # In appendix B the picture assumes a downsample factor of 4 instead of 8.

        up_mask = up_mask.view(batch_size, 1, 9, 8, 8, h, w)
        up_mask = torch.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

        upsampled_flow = F.unfold(8 * flow, kernel_size=3, padding=1).view(batch_size, 2, 9, 1, 1, h, w)
        upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)

        return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, 2, new_h, new_w)

    def forward(self, image1, image2, num_flow_updates=12):

        batch_size, _, h, w = image1.shape
        torch._assert((h, w) == image2.shape[-2:], "input images should have the same shape")
        torch._assert((h % 8 == 0) and (w % 8 == 0), "input image H and W should be divisible by 8")

        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        torch._assert(fmap1.shape[-2:] == (h / 8, w / 8), "The feature encoder should downsample H and W by 8")

        self.corr_block.build_pyramid(fmap1, fmap2)

        context_out = self.context_encoder(image1)
        torch._assert(context_out.shape[-2:] == (h / 8, w / 8), "The context encoder should downsample H and W by 8")

        # As in the original paper, the actual output of the context encoder is split in 2 parts:
        # - one part is used to initialize the hidden state of the reccurent units of the update block
        # - the rest is the "actual" context.
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size
        torch._assert(
            out_channels_context > 0,
            f"The context encoder outputs {context_out.shape[1]} channels, but it should have at least "
            f"hidden_state={hidden_state_size} channels",
        )
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = torch.relu(context)

        coords0 = make_coords_grid(batch_size, h // 8, w // 8).cuda()
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).cuda()

        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)

            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

            coords1 = coords1 + delta_flow

            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            upsampled_flow = self._upsample_flow(flow=(coords1 - coords0), up_mask=up_mask)
            flow_predictions.append(upsampled_flow)

        return flow_predictions
