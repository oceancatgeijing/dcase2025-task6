git status
git add .
git commit -m "基础模型训练完成，尝试融合架构"
git push

# 设置学术环境
source /etc/network_turbo

# 新建一个会话（名字可自定）
tmux new -s train

# 在 tmux 里正常激活环境、跑训练
conda activate d25_final
cd /root/autodl-tmp/dcase2025
python -m d25_t6.train --audiocaps --data_path=data --batch_size=32 ...

# 断线后：在本地重新 SSH 登录服务器，执行
tmux attach -t train
# 或简写
tmux a -t train

只断开连接，让程序继续跑（推荐）
按键盘：
Ctrl+b，松手后再按 d
会话会留在后台，训练继续。之后用 tmux attach -t <会话名> 可以再连回去。