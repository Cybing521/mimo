"""
Reusable Weights & Biases helpers.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
DEFAULT_KEY_PATH = os.path.expanduser("~/.config/mimo/wandb_api_key")
_LOADED_DOTENVS = set()


def _load_dotenv(path: Optional[str]) -> None:
    """Load key-value pairs from a .env file into os.environ (without overriding)."""
    if not path:
        return
    expanded = os.path.expanduser(path)
    if expanded in _LOADED_DOTENVS or not os.path.exists(expanded):
        return
    try:
        with open(expanded, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ and value:
                    os.environ[key] = value
        _LOADED_DOTENVS.add(expanded)
    except OSError:
        pass


def _read_key_file(path: str) -> Optional[str]:
    """Read API key from file."""
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return None
    try:
        with open(expanded, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content or None
    except OSError:
        return None


def ensure_wandb_api_key(
    explicit_key: Optional[str] = None,
    key_file: Optional[str] = None,
    dotenv_path: Optional[str] = DEFAULT_DOTENV_PATH,
):
    """
    Ensure WANDB_API_KEY is available in environment.
    Priority: env (possibly loaded from .env) > explicit key > key file > fallback file.
    """
    _load_dotenv(dotenv_path)
    if os.environ.get("WANDB_API_KEY"):
        return
    if explicit_key:
        os.environ["WANDB_API_KEY"] = explicit_key.strip()
        return
    candidate_files = [key_file, DEFAULT_KEY_PATH]
    for path in candidate_files:
        key_from_file = _read_key_file(path) if path else None
        if key_from_file:
            os.environ["WANDB_API_KEY"] = key_from_file.strip()
            return


def init_wandb(
    enabled: bool,
    project: Optional[str],
    entity: Optional[str],
    run_name: Optional[str],
    mode: str,
    config: Dict,
    tags: Optional[Iterable[str]] = None,
    group: Optional[str] = None,
    run_dir: Optional[str] = None,
):
    """
    Initialize a Weights & Biases run if enabled.
    """
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "未找到wandb库，请先执行 `pip install wandb` 后再开启在线可视化。"
        ) from exc

    init_kwargs = {
        "project": project or "ma-mimo",
        "entity": entity or None,
        "name": run_name,
        "config": config,
        "mode": mode,
        "tags": list(tags) if tags else None,
        "group": group,
        "dir": run_dir,
        "reinit": True,
    }
    # 移除值为 None 的键，避免 wandb 报错
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)


def log_metrics(run, metrics: Dict, step: Optional[int] = None):
    """
    Safe wandb.log wrapper.
    """
    if run is None or not metrics:
        return
    run.log(metrics, step=step)


def log_image(run, key: str, image_path: str, caption: Optional[str] = None, step: Optional[int] = None):
    """
    Log a matplotlib figure or any image to wandb.
    """
    if run is None or not os.path.exists(image_path):
        return
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    run.log({key: wandb.Image(image_path, caption=caption)}, step=step)


def log_line_series(
    run,
    x_values: List[float],
    series_dict: Dict[str, List[float]],
    title: str,
    x_name: str,
    key: str,
):
    """
    Log multiple line series curves to wandb dashboards.
    使用 wandb.Table 和 line_series 绘图（兼容不同版本 API）。
    """
    if run is None or not x_values or not series_dict:
        return
    try:
        import wandb  # type: ignore
    except ImportError:
        return

    columns = [x_name] + list(series_dict.keys())
    data = []
    num_points = len(x_values)
    for idx in range(num_points):
        row = [float(x_values[idx])]
        for series in series_dict.values():
            if idx < len(series):
                row.append(float(series[idx]))
            else:
                row.append(float("nan"))
        data.append(row)

    table = wandb.Table(columns=columns, data=data)
    
    # 尝试不同的 API 版本
    try:
        # 方法1：新版本 API（使用 xs 和 keys）
        plot = wandb.plot.line_series(
            xs=table,
            keys=list(series_dict.keys()),
            title=title,
            xname=x_name,
        )
        run.log({key: plot})
    except (TypeError, AttributeError) as e1:
        try:
            # 方法2：使用 table 作为第一个参数，keys 指定要绘制的列
            plot = wandb.plot.line_series(
                table,
                keys=list(series_dict.keys()),
                title=title,
                xname=x_name,
            )
            run.log({key: plot})
        except (TypeError, AttributeError) as e2:
            # 方法3：如果都失败，使用最简单的方式：直接记录为标量序列
            # wandb 会自动生成图表，这是最兼容的方式
            for idx, x_val in enumerate(x_values):
                log_dict = {}
                for series_name, series_values in series_dict.items():
                    if idx < len(series_values):
                        log_dict[f"{key}/{series_name}"] = series_values[idx]
                if log_dict:
                    run.log(log_dict, step=int(x_val))

