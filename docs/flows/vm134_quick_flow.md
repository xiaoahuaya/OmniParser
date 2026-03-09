# VM134 操作流程（精简版）

目标：在宿主机控制 `192.168.31.134`（VM），并在 VM 浏览器打开小红书。

## 1) VM 侧准备（只做一次）

```powershell
# VM(134) 管理员 PowerShell
winrm quickconfig -q
Enable-PSRemoting -Force
Set-Service WinRM -StartupType Automatic
Start-Service WinRM
New-NetFirewallRule -DisplayName "Allow WinRM 5985" -Direction Inbound -Protocol TCP -LocalPort 5985 -Action Allow
```

## 2) 宿主机连通验证

```powershell
Test-WSMan 192.168.31.134
```

## 3) VM 控制服务启动（在 VM 桌面里执行）

说明：必须在 VM 已登录桌面会话中启动，窗口不要关闭。

```powershell
& "C:\Program Files\Python311\python.exe" "C:\Users\37417\script\main.py" --port 5000
```

## 4) 宿主机启动 OmniParser + Gradio

```powershell
# 终端A
C:\Users\mateng\.conda\envs\OmniParser\python.exe F:\ideacode\OmniParser\omnitool\omniparserserver\omniparserserver.py

# 终端B
C:\Users\mateng\.conda\envs\OmniParser\python.exe F:\ideacode\OmniParser\omnitool\gradio\app.py --windows_host_url 192.168.31.134:5000 --omniparser_server_url localhost:9000
```

访问：

```text
http://127.0.0.1:7888
```

## 5) 快速验证“确实在控制 VM”

```powershell
# 宿主机：让 VM 打开小红书
$sec = ConvertTo-SecureString "123456" -AsPlainText -Force
$cred = New-Object pscredential("192.168.31.134\omniadmin",$sec)
Invoke-Command -ComputerName 192.168.31.134 -Credential $cred -ScriptBlock {
  Start-Process "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "https://www.xiaohongshu.com"
}
```

如果 VM 屏幕上打开了小红书，就说明链路是宿主机 -> VM。

## 6) 常见问题（最短定位）

1. `WinRM access denied`：本机 PowerShell 非管理员，或凭据错误。  
2. `/probe` 通但 `/screenshot` 500：VM 不是交互桌面会话（锁屏/后台会话）。  
3. Gradio 里还是宿主机画面：`ComputerTool` 仍在本地模式或 VM 服务未重启到新代码。  
4. `DeprecationWarning`：仅警告，不影响运行。  

