import pytest

from rl3dsr.validation.decoded_space import strict_3d_quality_verdict


def quality_rows(seeds=(1201,)):
    values = {'correct': (29., .95, .08), 'bicubic': (28., .93, .10),
              'shuffled': (26., .90, .15), 'disabled': (20., .80, .3)}
    return [dict(seed=s, item='chair', position=v, condition=c,
                 psnr=m[0], ssim=m[1], lpips=m[2])
            for s in seeds for v in range(4) for c, m in values.items()]


def test_development_gate_does_not_require_four_seeds():
    rows = quality_rows()
    assert strict_3d_quality_verdict(rows, development=True)['passed']
    assert not strict_3d_quality_verdict(rows)['passed']


def test_final_gate_and_duplicate_missing_view_validation():
    rows = quality_rows((2201, 2202, 2203, 2204))
    assert strict_3d_quality_verdict(rows)['passed']
    with pytest.raises(ValueError, match='duplicate'):
        strict_3d_quality_verdict(rows + [rows[0]])
    with pytest.raises(ValueError, match='four views'):
        strict_3d_quality_verdict([r for r in rows if r['position'] != 3])
    with pytest.raises(ValueError, match='finite'):
        strict_3d_quality_verdict([{**rows[0], 'psnr': float('nan')}, *rows[1:]])


def test_development_gate_preserves_margins():
    rows = quality_rows()
    for r in rows:
        if r['condition'] == 'correct':
            r['ssim'] = .931
    result = strict_3d_quality_verdict(rows, development=True)
    assert not result['passed']
    assert not result['checks']['bicubic_margin']
