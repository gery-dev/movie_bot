#!/usr/bin/env bash
# Выполняется НА СЕРВЕРЕ из папки бота (его вызывает deploy.sh). Ubuntu/Debian, нужен sudo.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

if ! command -v python3 >/dev/null || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "Нужен Python 3.11+"; exit 1
fi
# python3-venv — виртуальное окружение; libgomp1 — OpenMP для implicit (ALS) и scikit-learn
for pkg in python3-venv libgomp1; do
    dpkg -s "$pkg" >/dev/null 2>&1 || sudo apt-get install -y -qq "$pkg"
done

[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
# torch без видеокарты — иначе pip скачает сборку с CUDA (~2 ГБ лишнего)
venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -q -r requirements-bot.txt

mkdir -p logs data/backups
for unit in movie-bot.service movie-bot-backup.service movie-bot-backup.timer; do
    sed "s|__USER__|$USER|g; s|__DIR__|$DIR|g" "deploy/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now movie-bot-backup.timer
sudo systemctl enable movie-bot.service
sudo systemctl restart movie-bot.service
sleep 5
systemctl --no-pager --lines=5 status movie-bot.service || true
