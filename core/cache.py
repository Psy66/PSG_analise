# core/cache.py
import os
import json
import hashlib
import time
import pickle
from typing import Any, Optional

from core.config import CACHE_API_DIR, CACHE_TTL_SECONDS

os.makedirs(CACHE_API_DIR, exist_ok=True)


def _get_cache_key(api_url: str, endpoint: str, params: dict) -> str:
    """
    Генерирует уникальный ключ кэша на основе URL, endpoint и параметров.
    Параметры сортируются и преобразуются в строку для воспроизводимости.
    """
    params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
    combined = f"{api_url}|{endpoint}|{params_str}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def get_cached_data(
    api_url: str,
    endpoint: str,
    params: dict,
    ttl: Optional[int] = None
) -> Optional[Any]:
    """
    Возвращает данные из кэша, если они есть и не устарели.
    Если ttl не задан, используется CACHE_TTL_SECONDS из конфига.
    """
    if ttl is None:
        ttl = CACHE_TTL_SECONDS
    cache_key = _get_cache_key(api_url, endpoint, params)
    cache_path = os.path.join(CACHE_API_DIR, f"{cache_key}.pkl")
    if not os.path.exists(cache_path):
        return None
    mtime = os.path.getmtime(cache_path)
    if time.time() - mtime > ttl:
        # Устарело – удаляем
        os.remove(cache_path)
        return None
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cached_data(
    api_url: str,
    endpoint: str,
    params: dict,
    data: Any
) -> None:
    """Сохраняет данные в кэш."""
    cache_key = _get_cache_key(api_url, endpoint, params)
    cache_path = os.path.join(CACHE_API_DIR, f"{cache_key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)