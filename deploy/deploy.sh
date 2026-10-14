#!/usr/bin/env bash
# Развёртывание бота на сервер. Запускать на Mac из папки проекта:
#   ./deploy/deploy.sh user@host                # обновить код (данные пользователей на сервере не трогаются)
#   ./deploy/deploy.sh user@host --first        # первый раз: ещё и БД без пользователей, эмбеддинги, MovieLens small, .env
#   SSH_KEY=~/.ssh/ключ ./deploy/deploy.sh …     # если нужен конкретный SSH-ключ
# Перед первым развёртыванием остановите бота на Mac: два бота с одним токеном мешают друг другу.
# БД на сервер уходит «с нуля» (python -m db.clean_copy): фильмы и кэш TMDB, без данных пользователей.
set -euo pipefail
HOST="${1:?укажите user@host}"
FIRST="${2:-}"
DEST=/opt/movie_bot
# в bash 3.2 (стоит на macOS) пустой массив при set -u — ошибка, поэтому вместо массива — строка
SSH_OPTS=""
[ -n "${SSH_KEY:-}" ] && SSH_OPTS="-i $SSH_KEY -o IdentitiesOnly=yes"
ssh() { command ssh $SSH_OPTS "$@"; }
scp() { command scp $SSH_OPTS "$@"; }
RSYNC_SSH="ssh $SSH_OPTS"

ssh "$HOST" "command -v rsync >/dev/null || sudo apt-get install -y -qq rsync; sudo mkdir -p $DEST && sudo chown \$(whoami) $DEST"
rsync -az --delete -e "$RSYNC_SSH" \
    --exclude venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
    --exclude 'data/' --exclude 'logs/' --exclude '.env' --exclude 'reports/' --exclude 'notebooks/' \
    ./ "$HOST:$DEST/"

if [ "$FIRST" = "--first" ]; then
    ssh "$HOST" "mkdir -p $DEST/data/raw/movielens $DEST/data/processed"
    venv/bin/python -m db.clean_copy
    rsync -az -e "$RSYNC_SSH" data/movies-clean.db "$HOST:$DEST/data/movies.db"
    rsync -az -e "$RSYNC_SSH" data/processed/ "$HOST:$DEST/data/processed/"
    rsync -az -e "$RSYNC_SSH" data/raw/movielens/ml-latest-small "$HOST:$DEST/data/raw/movielens/"
    scp .env "$HOST:$DEST/.env"
    ssh "$HOST" "chmod 600 $DEST/.env"
fi

ssh "$HOST" "bash $DEST/deploy/setup_server.sh"
