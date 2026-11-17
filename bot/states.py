from aiogram.fsm.state import State, StatesGroup


class Onboarding(StatesGroup):
    genres = State()
    films = State()
    series = State()
    people = State()


class Calibration(StatesGroup):
    rating = State()


class Today(StatesGroup):
    asking = State()


class Review(StatesGroup):
    text = State()


class FavoriteSearch(StatesGroup):
    query = State()


class LetterboxdImport(StatesGroup):
    file = State()
