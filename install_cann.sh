#!/bin/bash
# CANN 安装脚本
# 使用方法: sudo bash install_cann.sh

set -e

echo "=========================================="
echo "CANN 安装脚本"
echo "=========================================="

# 1. 将 ht 用户添加到 HwHiAiUser 组
echo ""
echo "[1/5] 将 ht 用户添加到 HwHiAiUser 组..."
usermod -aG HwHiAiUser ht
echo "完成!"

# 2. 配置 Ascend 源
echo ""
echo "[2/5] 配置 Ascend 软件源..."
curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
yum makecache
echo "完成!"

# 3. 安装依赖
echo ""
echo "[3/5] 安装依赖包..."
yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
echo "完成!"

# 4. 安装 NPU 驱动 (如果需要更新)
echo ""
echo "[4/5] 检查 NPU 驱动..."
if rpm -q Ascend-hdk-910-npu-driver &>/dev/null; then
    echo "NPU 驱动已安装: $(rpm -q Ascend-hdk-910-npu-driver)"
else
    echo "安装 NPU 驱动..."
    yum install -y Ascend-hdk-910-npu-driver-25.5.0
fi
echo "完成!"

# 5. 安装 CANN Toolkit
echo ""
echo "[5/5] 安装 CANN Toolkit 和算子库..."
yum install -y Ascend-cann-toolkit-9.0.0_beta.2
yum install -y Ascend-cann-910-ops-9.0.0_beta.2
echo "完成!"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "请执行以下步骤完成配置:"
echo ""
echo "1. 重新登录或执行: newgrp HwHiAiUser"
echo ""
echo "2. 配置环境变量:"
echo "   source /usr/local/Ascend/cann/set_env.sh"
echo ""
echo "3. 验证安装:"
echo "   python3 -c \"import acl;print(acl.get_soc_name())\""
echo ""
