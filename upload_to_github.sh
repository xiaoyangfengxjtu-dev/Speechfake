#!/bin/bash

# GitHub 上传脚本
# 使用方法: ./upload_to_github.sh YOUR_USERNAME REPOSITORY_NAME

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <GitHub用户名> <仓库名称>"
    echo "示例: $0 yourusername speechfake-baselines"
    exit 1
fi

USERNAME=$1
REPO_NAME=$2
REPO_URL="https://github.com/$USERNAME/$REPO_NAME.git"

echo "=== GitHub 上传脚本 ==="
echo "用户名: $USERNAME"
echo "仓库名: $REPO_NAME"
echo "仓库URL: $REPO_URL"
echo ""

# 检查当前状态
echo "1. 检查当前 Git 状态..."
git status --porcelain
if [ $? -ne 0 ]; then
    echo "错误: 当前目录不是 Git 仓库"
    exit 1
fi

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "警告: 有未提交的更改，请先提交或暂存"
    echo "运行: git add . && git commit -m 'your message'"
    exit 1
fi

echo "✓ Git 状态正常"
echo ""

# 更新远程仓库地址
echo "2. 更新远程仓库地址..."
git remote set-url origin $REPO_URL
if [ $? -eq 0 ]; then
    echo "✓ 远程仓库地址已更新"
else
    echo "错误: 无法更新远程仓库地址"
    exit 1
fi

# 验证远程地址
echo "3. 验证远程仓库配置..."
git remote -v
echo ""

# 推送代码
echo "4. 推送代码到 GitHub..."
git push -u origin main
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 代码已成功推送到 GitHub!"
    echo "仓库地址: https://github.com/$USERNAME/$REPO_NAME"
    echo ""
    echo "下一步:"
    echo "1. 访问 https://github.com/$USERNAME/$REPO_NAME 确认上传"
    echo "2. 在服务器上克隆: git clone $REPO_URL"
    echo "3. 按照 SERVER_DEPLOYMENT.md 进行服务器部署"
else
    echo ""
    echo "错误: 推送失败"
    echo "可能的原因:"
    echo "1. GitHub 仓库不存在 - 请先在 GitHub 上创建仓库"
    echo "2. 权限问题 - 检查 GitHub 访问权限"
    echo "3. 网络问题 - 检查网络连接"
    echo ""
    echo "手动步骤:"
    echo "1. 访问 https://github.com/new 创建仓库 '$REPO_NAME'"
    echo "2. 确保仓库是空的 (不要初始化 README)"
    echo "3. 重新运行此脚本"
    exit 1
fi
