-- Схема БД бота. Фильмы и люди заполняются из IMDb (data_pipeline/imdb.py),
-- поля TMDB (tmdb_id, страна, язык, описание, постер) дополняются позже.

-- Фильмы и люди --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS movies (
    tconst          TEXT PRIMARY KEY,      -- IMDb ID, например tt0816692
    title           TEXT NOT NULL,         -- основное название
    original_title  TEXT,
    title_ru        TEXT,                  -- русское название из title.akas
    year            INTEGER,
    runtime         INTEGER,               -- минуты
    genres          TEXT,                  -- через запятую, как в IMDb
    rating          REAL,
    votes           INTEGER,
    tmdb_id         INTEGER,
    country         TEXT,
    language        TEXT,
    overview_ru     TEXT,
    poster_path     TEXT,
    kp_id           INTEGER                -- ID Кинопоиска (из Wikidata)
);
CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);
CREATE INDEX IF NOT EXISTS idx_movies_votes ON movies(votes);

CREATE TABLE IF NOT EXISTS movie_genres (
    tconst  TEXT NOT NULL REFERENCES movies(tconst),
    genre   TEXT NOT NULL,
    PRIMARY KEY (tconst, genre)
);
CREATE INDEX IF NOT EXISTS idx_movie_genres_genre ON movie_genres(genre);

CREATE TABLE IF NOT EXISTS people (
    nconst      TEXT PRIMARY KEY,          -- IMDb ID, например nm0634240
    name        TEXT NOT NULL,
    birth_year  INTEGER,
    death_year  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);

CREATE TABLE IF NOT EXISTS movie_people (
    tconst  TEXT NOT NULL REFERENCES movies(tconst),
    nconst  TEXT NOT NULL REFERENCES people(nconst),
    role    TEXT NOT NULL,                 -- actor | director
    ord     INTEGER,                       -- порядок в титрах (для актёров)
    PRIMARY KEY (tconst, nconst, role)
);
CREATE INDEX IF NOT EXISTS idx_movie_people_nconst ON movie_people(nconst);

-- Пользователи и их данные ---------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY,   -- Telegram ID
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    onboarding_done INTEGER NOT NULL DEFAULT 0,
    calibration_done INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    tconst      TEXT NOT NULL REFERENCES movies(tconst),
    rating      INTEGER NOT NULL,          -- 1–5
    source      TEXT NOT NULL,             -- calibration | recommendation | letterboxd
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tconst)
);

CREATE TABLE IF NOT EXISTS reactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    tconst      TEXT NOT NULL REFERENCES movies(tconst),
    reaction    TEXT NOT NULL,             -- not_seen | skip | seen | not_interested | later
    review      TEXT,
    sentiment   REAL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    feature     TEXT NOT NULL,             -- например genre:Drama, director:nm0634240, decade:1990
    weight      REAL NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, feature)
);

CREATE TABLE IF NOT EXISTS daily_session (
    user_id     INTEGER PRIMARY KEY REFERENCES users(user_id),
    answers     TEXT NOT NULL,             -- JSON ответов опроса «на сегодня»
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_lists (
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    tconst      TEXT NOT NULL REFERENCES movies(tconst),
    list        TEXT NOT NULL,             -- seen | planned | hidden
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tconst, list)
);

CREATE TABLE IF NOT EXISTS favorites (
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    nconst      TEXT NOT NULL REFERENCES people(nconst),
    added_by    TEXT NOT NULL,             -- manual | suggestion
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, nconst)
);

CREATE TABLE IF NOT EXISTS groups (
    group_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    INTEGER NOT NULL REFERENCES users(user_id),
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- имя из Telegram — только чтобы показать участникам компании (/together), кто с ними
CREATE TABLE IF NOT EXISTS user_names (
    user_id     INTEGER PRIMARY KEY REFERENCES users(user_id),
    name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id    INTEGER NOT NULL REFERENCES groups(group_id),
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    tconst      TEXT NOT NULL REFERENCES movies(tconst),
    shown_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    asked_at    TEXT,                      -- когда бот спросил, посмотрел ли
    answer      TEXT,                      -- rated | seen | not_interested | later | hidden | ignored
    mode        TEXT                       -- recommend | today | query
);
CREATE INDEX IF NOT EXISTS idx_recommendations_open ON recommendations(answer, shown_at);

-- ответы на опрос «насколько подборки подходят» (онлайн-тест, этап 6)
CREATE TABLE IF NOT EXISTS survey (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    wave        INTEGER NOT NULL,          -- после скольких оценок из подборок спросили
    score       INTEGER NOT NULL,          -- 1–5
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Кэш внешних API ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS api_cache (
    source      TEXT NOT NULL,             -- tmdb | omdb
    key         TEXT NOT NULL,             -- путь запроса с параметрами
    response    TEXT NOT NULL,             -- JSON
    fetched_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, key)
);
