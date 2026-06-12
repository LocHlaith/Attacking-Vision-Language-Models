import numpy as np

from avlm_or.text_attack import (
    TextTemplate,
    efficient_linearized_milp,
    manual_linearized_selection,
    render_text_template,
)


def small_template() -> TextTemplate:
    mask = np.zeros((224, 224), dtype=bool)
    stroke = {(0, 0), (0, 1), (0, 2)}
    for vertex in stroke:
        mask[vertex] = True
    return TextTemplate("small", mask, [stroke], [(0, 0)])


def test_manual_linearized_selection() -> None:
    template = small_template()
    gain = np.zeros_like(template.mask, dtype=float)
    gain[template.mask] = 1.0
    result = manual_linearized_selection(template, gain, 2.0, 0.5)
    assert result is not None
    assert int(result.sum()) == 2


def test_efficient_linearized_milp_with_network_flow() -> None:
    template = small_template()
    gain = np.zeros_like(template.mask, dtype=float)
    gain[template.mask] = 1.0
    result = efficient_linearized_milp(template, gain, 2.0, 0.5, time_limit=10.0)
    assert result is not None
    assert int(result.sum()) == 2


def test_rendered_text_rotation_and_size() -> None:
    small = render_text_template(font_size=20, angle=0)
    rotated = render_text_template(font_size=20, angle=90)
    assert small.mask.any()
    assert rotated.mask.any()
    assert small.name == "20_center_+0deg"
    assert rotated.name == "20_center_+90deg"
    assert not np.array_equal(small.mask, rotated.mask)
