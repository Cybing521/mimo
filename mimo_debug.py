"""
MIMO调试版本 - 用于诊断死机问题
"""
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import time
import sys

def test_cvx_solver():
    """测试CVX求解器是否正常"""
    print("测试CVX求解器...")
    try:
        Q_var = cp.Variable((4, 4), hermitian=True)
        H_r = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        sigma = 1.0
        power = 10.0
        
        obj = cp.log_det(np.eye(4) + (1/sigma) * H_r @ Q_var @ H_r.T.conj())
        constraints = [cp.trace(Q_var) <= power, Q_var >> 0]
        prob = cp.Problem(cp.Maximize(obj), constraints)
        
        start = time.time()
        prob.solve(solver=cp.SCS, verbose=True, max_iters=100, eps=1e-3)
        elapsed = time.time() - start
        
        print(f"CVX求解完成，耗时: {elapsed:.2f}秒")
        print(f"状态: {prob.status}")
        return True
    except Exception as e:
        print(f"CVX求解失败: {e}")
        return False

def test_single_iteration():
    """测试单次迭代"""
    print("\n测试单次完整迭代...")
    from mimo_exact_reproduction import run_single_trial
    
    try:
        args = (0, 2, 10, 1.0)  # trial=0, A_lambda=2, Lr=10, lambda_val=1.0
        start = time.time()
        result = run_single_trial(args)
        elapsed = time.time() - start
        
        print(f"单次迭代完成，耗时: {elapsed:.2f}秒")
        print(f"结果: {result}")
        return True
    except Exception as e:
        print(f"单次迭代失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("MIMO调试诊断工具")
    print("="*60)
    
    # 测试1: CVX求解器
    if not test_cvx_solver():
        print("\n⚠️ CVX求解器有问题！")
        return
    
    # 测试2: 单次迭代
    if not test_single_iteration():
        print("\n⚠️ 单次迭代执行失败！")
        return
    
    print("\n✓ 所有测试通过！")
    print("\n建议:")
    print("1. 先用少量试验测试: python mimo_exact_reproduction.py 10")
    print("2. 监控内存使用情况")
    print("3. 考虑减少并行进程数")

if __name__ == "__main__":
    main()
