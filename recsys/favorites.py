"""Любимые актёры и режиссёры: поиск, добавление, автопредложение, непросмотренные фильмы."""
from recsys.profile import ensure_user, excluded_films

SUGGEST_MIN_FILMS = 3   # предлагать человека, если высоко оценено столько его фильмов
SUGGEST_MIN_RATING = 4


def search_people(conn, query: str, limit: int = 10) -> list[dict]:
    """Поиск по имени (имена в IMDb латиницей). Первыми — те, у кого больше известных фильмов."""
    rows = conn.execute(
        """SELECT p.nconst, p.name, group_concat(DISTINCT mp.role) AS roles, count(DISTINCT mp.tconst) AS films,
                  sum(m.votes) AS fame
           FROM people p JOIN movie_people mp USING (nconst) JOIN movies m USING (tconst)
           WHERE p.name LIKE ? GROUP BY p.nconst ORDER BY fame DESC LIMIT ?""",
        (f"%{query.strip()}%", limit),
    ).fetchall()
    return [{"nconst": r[0], "name": r[1], "roles": r[2].split(","), "films": r[3]} for r in rows]


def add_favorite(conn, user_id: int, nconst: str, added_by: str = "manual") -> None:
    """added_by: manual — сам добавил, suggestion — согласился на предложение, declined — отказался."""
    ensure_user(conn, user_id)
    conn.execute(
        """INSERT INTO favorites (user_id, nconst, added_by) VALUES (?, ?, ?)
           ON CONFLICT(user_id, nconst) DO UPDATE SET added_by = excluded.added_by""",
        (user_id, nconst, added_by),
    )


def remove_favorite(conn, user_id: int, nconst: str) -> None:
    conn.execute("DELETE FROM favorites WHERE user_id = ? AND nconst = ?", (user_id, nconst))


def favorite_people(conn, user_id: int) -> set[str]:
    rows = conn.execute("SELECT nconst FROM favorites WHERE user_id = ? AND added_by != 'declined'", (user_id,))
    return {r[0] for r in rows}


def suggest_favorites(conn, user_id: int) -> list[dict]:
    """Люди, чьи фильмы пользователь высоко оценил не меньше SUGGEST_MIN_FILMS раз, ещё не в избранном
    и не отклонённые раньше: «вы высоко оценили 3 фильма Нолана, добавить в любимые?»"""
    rows = conn.execute(
        """SELECT mp.nconst, p.name, mp.role, count(DISTINCT r.tconst) AS liked
           FROM ratings r JOIN movie_people mp USING (tconst) JOIN people p USING (nconst)
           WHERE r.user_id = ? AND r.rating >= ?
             AND mp.nconst NOT IN (SELECT nconst FROM favorites WHERE user_id = ?)
           GROUP BY mp.nconst, mp.role HAVING liked >= ?
           ORDER BY liked DESC, mp.role DESC""",
        (user_id, SUGGEST_MIN_RATING, user_id, SUGGEST_MIN_FILMS),
    ).fetchall()
    seen, result = set(), []
    for nconst, name, role, liked in rows:  # человек может быть и режиссёром, и актёром — берём одну строку
        if nconst not in seen:
            seen.add(nconst)
            result.append({"nconst": nconst, "name": name, "role": role, "liked": liked})
    return result


def person_unseen(conn, user_id: int, nconst: str, limit: int = 10) -> list[dict]:
    """Фильмы человека, которых пользователь не видел, лучшие по рейтингу — первыми."""
    skip = excluded_films(conn, user_id)
    rows = conn.execute(
        """SELECT DISTINCT m.tconst, coalesce(m.title_ru, m.title), m.year, m.rating
           FROM movie_people mp JOIN movies m USING (tconst)
           WHERE mp.nconst = ? ORDER BY m.rating DESC, m.votes DESC""",
        (nconst,),
    ).fetchall()
    films = [{"tconst": t, "title": title, "year": y, "rating": r} for t, title, y, r in rows if t not in skip]
    return films[:limit]
