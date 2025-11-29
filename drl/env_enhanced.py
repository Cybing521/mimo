"""
增强版环境：支持智能初始化和课程学习
====================================
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .env import MAMIMOEnv
from core.mimo_core import MIMOSystem


class EnhancedMAMIMOEnv(MAMIMOEnv):
    """
    增强版环境，支持：
    1. 智能初始化（使用AO结果）
    2. 课程学习（动态调整难度）
    3. 更好的探索策略
    """
    
    def __init__(
        self,
        use_ao_init: bool = False,
        curriculum_learning: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_ao_init = use_ao_init
        self.curriculum_learning = curriculum_learning
        self.ao_system = None
        
        if use_ao_init:
            # 创建AO系统用于初始化
            self.ao_system = MIMOSystem(
                N=self.N, M=self.M,
                Lt=self.Lt, Lr=self.Lr,
                SNR_dB=self.SNR_dB
            )
    
    def reset(
        self,
        episode: int = 0,
        total_episodes: int = 5000,
        ao_positions: Optional[Dict] = None
    ) -> np.ndarray:
        """
        增强的reset方法
        
        Args:
            episode: 当前episode编号（用于课程学习）
            total_episodes: 总episode数
            ao_positions: 可选的AO初始化位置
        """
        # 课程学习：动态调整难度
        if self.curriculum_learning:
            config = self._get_curriculum_config(episode, total_episodes)
            # 更新环境参数（需要重新创建MIMOSystem）
            # 这里简化处理，实际可以动态调整
            pass
        
        # 智能初始化
        if self.use_ao_init and ao_positions is None:
            # 运行一次快速AO获得初始位置
            try:
                result = self.ao_system.run_optimization(
                    self.A_lambda, mode='Proposed'
                )
                # 从结果中提取位置（需要修改MIMOSystem返回位置）
                # 这里简化，实际需要修改run_optimization返回位置
                ao_positions = {
                    'tx': None,  # 需要从result中提取
                    'rx': None
                }
            except:
                ao_positions = None
        
        if ao_positions is not None and self.use_ao_init:
            # 使用AO位置初始化
            self.t = ao_positions['tx'].copy()
            self.r = ao_positions['rx'].copy()
            
            # 添加小的随机扰动，保持探索
            noise_scale = 0.1 * self.mimo_system.D
            self.t += np.random.randn(*self.t.shape) * noise_scale
            self.r += np.random.randn(*self.r.shape) * noise_scale
            
            # 投影到可行域
            self.t = self._project_to_feasible_region(
                self.t, self.t, is_transmit=True
            )
            self.r = self._project_to_feasible_region(
                self.r, self.r, is_transmit=False
            )
        else:
            # 使用原始初始化
            super().reset()
            return self._get_state()
        
        # 重置其他变量
        self.current_step = 0
        self.capacity_history = []
        self._initialize_channel_params()
        self._update_channel()
        
        return self._get_state()
    
    def _get_curriculum_config(self, episode: int, total_episodes: int) -> Dict:
        """课程学习配置"""
        progress = episode / total_episodes
        
        if progress < 0.3:
            # 阶段1：简单情况
            return {
                'A_lambda': 2.0,
                'SNR_dB': 30.0,
                'max_steps': 30
            }
        elif progress < 0.6:
            # 阶段2：中等难度
            return {
                'A_lambda': 3.0,
                'SNR_dB': 25.0,
                'max_steps': 50
            }
        else:
            # 阶段3：困难情况
            return {
                'A_lambda': 4.0,
                'SNR_dB': 15.0,
                'max_steps': 100
            }

