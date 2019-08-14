import tensorflow as tf

from merlin.context import active_context


class ConvolutionPadding:
    """
    Abstract base class for custom convolution paddings.
    """

    def setup(self, conv: tf.Module):
        raise NotImplementedError

    def __call__(self, inputs):
        raise NotImplementedError


class SamePadding(ConvolutionPadding):
    """
    For strided convolutions, the TensorFlow "same" padding mode behaves
    asymmetrically depending on the input tensor shape. This can result in
    output feature maps being mis-aligned (in terms of the corners) by a pixel.
    The aligned mode explicitly zero-pads the tensor such that both even and odd
    shaped tensors behave similarly. Consequently, the output in aligned mode
    matches densely convolving without a stride and then subsampling with the given
    stride, regardless of the input tensor shape.

    aligned  : If true, explicitly pads the input if necessary as described above.
               Otherwise, a no-op that sets the associated convolution to use the
               default 'same' padding.
    """

    def __init__(self, aligned: bool):
        self.aligned = aligned
        self.padding = None

    def setup(self, conv):
        # Make sure the native padding won't interfere.
        assert conv.padding == 'valid'

        # Aligned padding is only necessary for non-unitary strides
        if self.aligned and any(stride > 1 for stride in conv.strides):
            self.padding = self.compute_padding(conv)
        else:
            # This convolution does not require explicit padding.
            # Use the native "same" padding mode.
            conv.padding = 'same'

    def compute_padding(self, conv):
        paddings = []
        for dim in range(2):
            # Compute the "effective" size of the convolution kernel, incorporating
            # any dilation rates (which expand the kernel's receptive field).
            kernel_size = conv.kernel_size[dim]
            dilation_rate = conv.dilation_rate[dim]
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
            # Compute the padding. If the padding cannot be evenly split, assign the extra
            # padding to the end.
            padding_total = effective_kernel_size - 1
            padding_start = padding_total // 2
            padding_end = padding_total - padding_start
            paddings.append([padding_start, padding_end])

        # Add the zero paddings along the batch and channel axes.
        if active_context.is_channels_first:
            return [[0, 0], [0, 0]] + paddings
        return [[0, 0]] + paddings + [[0, 0]]

    def __call__(self, inputs):
        if self.padding is None:
            return inputs
        return tf.pad(inputs, self.padding)
