Configure proxy and pip mirror settings for PPIO cluster nodes.

## Proxy

All nodes access external network via proxy on localhost:
```bash
export http_proxy=http://127.0.0.1:1083
export https_proxy=http://127.0.0.1:1083
export no_proxy=localhost,127.0.0.1,sealos.hub,10.0.0.0/8
```

### Usage scenarios

**curl/wget on host:**
```bash
https_proxy=http://127.0.0.1:1083 curl -L https://huggingface.co/...
```

**apt-get in pod (requires hostNetwork: true):**
```bash
export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083
apt-get update -qq && apt-get install -y -qq <packages>
unset http_proxy https_proxy
```

**pip install on host or in pod with hostNetwork:**
```bash
https_proxy=http://127.0.0.1:1083 pip install <package>
```

**containerd proxy (for crictl pull):**
Create `/etc/systemd/system/containerd.service.d/http-proxy.conf`:
```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:1083"
Environment="HTTPS_PROXY=http://127.0.0.1:1083"
Environment="NO_PROXY=localhost,127.0.0.1,sealos.hub,10.0.0.0/8"
```
Then: `systemctl daemon-reload && systemctl restart containerd`

## Pip Mirror (for China mainland nodes without proxy)

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn <package>
```

Alternative mirrors:
- Tsinghua: `https://pypi.tuna.tsinghua.edu.cn/simple/`
- USTC: `https://pypi.mirrors.ustc.edu.cn/simple/`
- Huawei: `https://repo.huaweicloud.com/repository/pypi/simple/`

Permanent config (`~/.pip/pip.conf`):
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
```

When helping the user with proxy or pip mirror setup, apply the appropriate configuration based on the scenario.
