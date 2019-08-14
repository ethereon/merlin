import tensorflow as tf

from merlin.context import DataOrdering, active_context
from merlin.ops.image import resample
from merlin.shape import TetraShape


def _test_scoped_resampled_shape(input_shape, output_shape, **resample_kwargs):
    input_img = tf.zeros(input_shape)
    output_img = resample(input_img, method=resample.NEAREST_NEIGHBOR, **resample_kwargs)
    assert output_img.shape == output_shape


def _get_resample_kwargs(resample_kwargs, is_nchw):
    if 'select' in resample_kwargs:
        return resample_kwargs['select'](is_nchw)
    return resample_kwargs


def _test_resampled_shape(input_shape, output_shape, **resample_kwargs):
    with active_context.replace(data_ordering=DataOrdering.NHWC):
        _test_scoped_resampled_shape(
            input_shape=input_shape,
            output_shape=output_shape,
            **_get_resample_kwargs(resample_kwargs, False)
        )

    with active_context.replace(data_ordering=DataOrdering.NCHW):
        _test_scoped_resampled_shape(
            input_shape=input_shape.nchw,
            output_shape=output_shape.nchw,
            **_get_resample_kwargs(resample_kwargs, True)
        )


def test_upsample_symmetric():
    _test_resampled_shape(
        input_shape=TetraShape(2, 12, 12, 3),
        output_shape=TetraShape(2, 24, 24, 3),
        scale=2
    )


def test_upsample_asymmetric():
    _test_resampled_shape(
        input_shape=TetraShape(2, 12, 12, 3),
        output_shape=TetraShape(2, 24, 36, 3),
        scale=(2, 3)
    )


def test_resize_like():
    def select(is_nchw):
        like_shape = TetraShape(1, 17, 16, 1)
        return dict(like=tf.zeros(like_shape.nchw if is_nchw else like_shape))

    _test_resampled_shape(
        input_shape=TetraShape(2, 12, 12, 3),
        output_shape=TetraShape(2, 17, 16, 3),
        select=select
    )


def test_resize_output_shape():
    _test_resampled_shape(
        input_shape=TetraShape(2, 12, 12, 3),
        output_shape=TetraShape(2, 20, 22, 3),
        size=(20, 22)
    )
