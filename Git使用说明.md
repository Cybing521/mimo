# Git 使用说明

## ⚠️ 重要提示：需要清理已提交的文件

您刚才执行了 `git commit`，但是 **虚拟环境目录 `venv/` 已经被提交到仓库中了**（约1475个文件）。

### 问题说明

`.gitignore` 文件只对**未跟踪的文件**有效。对于已经被 Git 跟踪（已提交）的文件，`.gitignore` 不会自动将它们移除。

### 解决方案

#### 方案1: 从Git中移除venv/（推荐）

```bash
# 1. 从Git索引中移除venv/，但保留本地文件
git rm -r --cached venv/

# 2. 提交这个移除操作
git commit -m "Remove venv/ from Git tracking"

# 3. 验证.gitignore生效
git status
# 应该看到venv/不再出现在未跟踪文件中
```

**说明**:
- `git rm -r --cached venv/`: 从Git索引移除，但不删除本地文件
- `-r`: 递归移除整个目录
- `--cached`: 只从Git索引移除，保留工作区文件

#### 方案2: 如果还没有push到远程仓库

如果您还没有执行 `git push`，可以修改最后一次提交：

```bash
# 1. 从Git索引移除venv/
git rm -r --cached venv/

# 2. 修改最后一次提交（而不是创建新提交）
git commit --amend -m "Initial commit (exclude venv)"

# 这样历史记录会更干净，就像venv/从未被提交过
```

#### 方案3: 如果已经push到远程仓库

如果已经推送到远程，**不建议强制修改历史**（除非您是唯一的开发者）。使用方案1即可。

---

## .gitignore 文件说明

### 核心界定原则

`.gitignore` 遵循以下界定原则：

1. **自动生成的文件** → 排除
   - 原因：可以重新生成，无需版本控制
   - 例如：`__pycache__/`, `*.pyc`

2. **环境相关的文件** → 排除
   - 原因：不同环境配置不同，不应共享
   - 例如：`venv/`, `.env`, IDE配置

3. **大型文件** → 排除
   - 原因：Git不适合存储大文件
   - 例如：数据集、模型文件

4. **敏感信息** → 排除
   - 原因：安全考虑
   - 例如：`.env`（可能含密钥）

5. **临时文件** → 排除
   - 原因：运行时产生，无需保存
   - 例如：`*.log`, `*.tmp`

6. **系统特定文件** → 排除
   - 原因：跨平台兼容性
   - 例如：`.DS_Store`（macOS）, `Thumbs.db`（Windows）

### 应该保留的文件

以下文件**应该**纳入版本控制：

- ✅ `*.py` - 源代码（核心）
- ✅ `*.m` - MATLAB代码（原始文件）
- ✅ `*.md` - 文档
- ✅ `requirements.txt` - 依赖列表
- ✅ `README.md` - 项目说明
- ✅ `.gitignore` - Git配置

可选保留：
- `*.png` - 结果图像（如果是重要实验结果）
- `*.npy` - 小型数据文件（如果是关键数据）

---

## 完整的Git工作流程

### 初始设置（一次性）

```bash
# 1. 初始化仓库（如果还没有）
git init

# 2. 添加.gitignore
git add .gitignore
git commit -m "Add .gitignore"

# 3. 清理已跟踪的不需要的文件
git rm -r --cached venv/
git commit -m "Remove venv/ from tracking"
```

### 日常工作流程

```bash
# 1. 查看修改状态
git status

# 2. 添加修改的文件
git add mimo_optimized.py          # 添加单个文件
# 或
git add *.py                       # 添加所有Python文件
# 或
git add .                          # 添加所有更改（谨慎使用）

# 3. 提交
git commit -m "Fix optimization bug"

# 4. 推送到远程（如果有）
git push origin main
```

### 检查.gitignore是否生效

```bash
# 查看哪些文件会被忽略
git status --ignored

# 或者测试特定文件
git check-ignore -v venv/

# 应该输出：.gitignore:33:venv/    venv/
# 表示该文件被.gitignore第33行规则忽略
```

---

## 常见问题

### Q1: 为什么venv/还在Git中？

**原因**: `.gitignore` 只对新文件有效，不影响已跟踪的文件。

**解决**: 使用 `git rm -r --cached venv/` 移除。

### Q2: 我想保留某些被.gitignore排除的文件怎么办？

**方法1**: 注释掉.gitignore中对应的行

```gitignore
# *.png  # 注释掉这行，保留PNG文件
```

**方法2**: 强制添加

```bash
git add -f important_result.png
```

### Q3: 如何查看.gitignore是否正确？

```bash
# 方法1: 查看未跟踪的文件
git status

# 方法2: 查看被忽略的文件
git status --ignored

# 方法3: 测试特定文件
git check-ignore -v venv/lib/python3.11/site-packages/numpy
```

### Q4: 我不小心提交了敏感信息怎么办？

**紧急处理**:

```bash
# 1. 从历史中完全删除文件（危险操作！）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# 2. 强制推送（如果已push）
git push origin --force --all
```

**注意**: 如果已经公开，应立即更换密钥/密码，因为Git历史可能已被他人克隆。

---

## 最佳实践

### 1. 提交前检查

```bash
# 总是在提交前检查将要提交什么
git status
git diff --staged
```

### 2. 有意义的提交信息

```bash
# ❌ 不好
git commit -m "update"

# ✅ 好
git commit -m "Fix quadprog solver instability in small regions"
```

### 3. 小而频繁的提交

```bash
# ✅ 每完成一个功能就提交
git commit -m "Add smart antenna initialization"
git commit -m "Implement robust position optimizer"
git commit -m "Add numerical stability checks"

# ❌ 一次提交所有更改
git commit -m "Complete all optimizations"
```

### 4. 分支管理（可选）

```bash
# 创建新分支进行实验
git checkout -b experiment/new-optimizer

# 完成后合并回主分支
git checkout main
git merge experiment/new-optimizer
```

---

## 立即执行的步骤

基于您刚才的提交，建议立即执行：

```bash
# 1. 确认.gitignore已添加
git add .gitignore
git commit -m "Add .gitignore file"

# 2. 移除venv/从Git跟踪
git rm -r --cached venv/

# 3. 提交移除操作
git commit -m "Remove venv/ from version control"

# 4. 验证状态
git status

# 5. 如果满意，推送到远程（如果有）
# git push origin main
```

执行完这些步骤后，您的仓库将变得干净，只包含必要的源代码和文档文件。
