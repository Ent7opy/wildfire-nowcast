# Task: Shared `pagination.py` Module

**Location:** `fires/repo.py:70-116` (source), `forecast/` routes (future consumers)
**Impact:** Low — clean code
**Maturity target:** `mvp_operational`

## Problem

`_encode_cursor`, `_decode_cursor`, and `_build_page` in `fires/repo.py:70-116` are private helpers with no obvious reason to live inside the fires module. When forecast routes need cursor pagination (likely soon), they will either duplicate this logic or create a circular import by pulling from `fires/repo.py`.

## Proposed Solution

Extract to a shared `api/pagination.py` module:

```python
# api/pagination.py
import base64, json
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None
    total: int | None = None  # optional, expensive to compute

def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": offset}).encode()).decode()

def decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    return int(data["offset"])

def build_page(items: list[T], offset: int, limit: int, has_more: bool) -> Page[T]:
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return Page(items=items, next_cursor=next_cursor)
```

`fires/repo.py` imports from `api.pagination` instead of defining its own private helpers.

## Acceptance Criteria

- [ ] `api/pagination.py` created with `encode_cursor`, `decode_cursor`, `build_page`, and `Page[T]`
- [ ] `fires/repo.py` imports from `api.pagination` — private `_encode_cursor` etc. removed
- [ ] `Page[T]` is a typed generic — no `dict` return types for paginated responses
- [ ] Unit tests for `encode_cursor` / `decode_cursor` round-trip
- [ ] Forecast routes that need pagination in future use `api.pagination` without any new duplication

## Notes

- Keep the cursor format opaque to callers — internal representation (offset vs. keyset) can change without breaking the API contract
- If keyset pagination is needed later (e.g. for detection_time-ordered queries), the `decode_cursor` function is the only place that needs updating
