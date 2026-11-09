"""Filtros de búsqueda en Milvus (idioma, colección de origen y fechas), compartidos por la API y el pipeline.

Los valores se validan antes de armar la expresión, así que un texto del usuario nunca se interpreta
como parte del filtro. Las fechas son días UTC con ambos extremos incluidos, igual que `ventana_fecha` del gold.
"""
import re
from datetime import date, datetime, timedelta, timezone

SOURCES = ('bluesky_posts', 'url_crawls')
LANGUAGE = re.compile(r'[a-z]{2,3}')

def _day(value):
    return value if isinstance(value, date) else date.fromisoformat(str(value))

def _epoch(day):
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())

def window_bounds(window):
    """(inicio, fin) en segundos UTC de una ventana {desde, hasta}; el fin es exclusivo (día siguiente a `hasta`)."""
    return _epoch(_day(window['desde'])), _epoch(_day(window['hasta']) + timedelta(days=1))

def window_filter(window):
    """Expresión de Milvus para una ventana {desde, hasta} sobre `created_at` (ambos días incluidos)."""
    start, end = window_bounds(window)
    return f'created_at >= {start} and created_at < {end}'

def build_filter(language=None, collection_name=None, desde=None, hasta=None):
    """Expresión de Milvus que combina los filtros dados ('' si no hay ninguno)."""
    parts = []
    if language:
        if not LANGUAGE.fullmatch(language): raise ValueError(f'Idioma no válido: {language!r} (código ISO de 2 o 3 letras)')
        parts.append(f'language == "{language}"')
    if collection_name:
        if collection_name not in SOURCES: raise ValueError(f'Colección de origen no válida: {collection_name!r}; opciones: {SOURCES}')
        parts.append(f'collection_name == "{collection_name}"')
    if desde and hasta and _day(desde) > _day(hasta): raise ValueError('La fecha inicial es posterior a la final')
    if desde: parts.append(f'created_at >= {_epoch(_day(desde))}')
    if hasta: parts.append(f'created_at < {_epoch(_day(hasta) + timedelta(days=1))}')
    return ' and '.join(parts)
