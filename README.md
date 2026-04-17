# YOLOv8 NPU

基于华为昇腾 NPU 的 YOLOv8 训练项目。

## 环境要求

- 华为昇腾 NPU 设备（Atlas 系列）
- CANN 9.0.0-beta.2

## CANN 安装指南

### 1. 配置用户属组

```bash
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

### 2. 安装依赖 & 配置源

```bash
sudo yum makecache
sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
```

### 3. 安装 NPU 驱动

```bash
sudo yum install -y Ascend-hdk-910-npu-driver-25.5.0
```

### 4. 安装 Toolkit

```bash
sudo yum install Ascend-cann-toolkit-9.0.0_beta.2
sudo yum install Ascend-cann-910-ops-9.0.0_beta.2
```

### 5. 验证安装

```bash
# 配置环境变量
source /usr/local/Ascend/cann/set_env.sh

# 验证 NPU 是否识别
python3 -c "import acl;print(acl.get_soc_name())"
```

若返回芯片型号，则安装成功。

## 运行测试

CANN 安装完成后，执行以下脚本测试 NPU 环境：

```bash
bash test_npu_env.sh
```

## 参考资料

- [CANN 快速开始 FAQ](https://www.hiascend.com/document)
- [CANN 安装指南](https://www.hiascend.com/document)