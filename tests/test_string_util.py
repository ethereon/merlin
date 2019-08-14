from merlin.util.string import camel_to_snake


def test_camel_to_snake():
    verification_mapping = {
        'Conv2D': 'conv_2d',
        'Conv2d': 'conv_2d',
        'Max2dPooling': 'max_2d_pooling',
        'Max2DPooling': 'max_2d_pooling',
        'ROIPooling': 'roi_pooling',
        'alpha': 'alpha',
        'Alpha': 'alpha',
        'AlphaBeta': 'alpha_beta',
        'AlphaBetaGamma': 'alpha_beta_gamma',
        'AlphaBetaGamma2': 'alpha_beta_gamma_2',
        'Alpha2BetaGamma': 'alpha_2_beta_gamma',
        '2AlphaBetaGamma': '2_alpha_beta_gamma',
        '2alphaBetaGamma': '2alpha_beta_gamma'
    }
    for camelcase, expected_snakecase in verification_mapping.items():
        assert camel_to_snake(camelcase) == expected_snakecase
