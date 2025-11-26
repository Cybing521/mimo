"""
基础单元测试：确保核心几何函数在重构前保持正确。
运行：pytest tests/unit
"""

import numpy as np

from core.mimo_core import MIMOSystem


def test_initialize_antennas_fixed_ula_spacing():
    """ULA 初始化应保持 lambda/2 等间距并居中。"""
    system = MIMOSystem(N=4, M=4, lambda_val=1.0)
    positions = system.initialize_antennas_fixed_ula(4)

    diffs = np.diff(positions[0])
    assert np.allclose(diffs, system.D)
    assert np.allclose(np.mean(positions[0]), 0.0, atol=1e-12)
    assert np.allclose(positions[1], 0.0)


def test_compute_field_response_unit_magnitude():
    """场响应的幅度应恒为 1。"""
    system = MIMOSystem()
    r = np.array([0.1, -0.2])
    theta = np.pi / 4
    phi = np.pi / 3

    response, phase = system.compute_field_response(r, theta, phi)

    assert isinstance(phase, float)
    assert np.isclose(np.abs(response), 1.0)

