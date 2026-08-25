import httpx
from app.core.config import settings
class LaravelClientError(Exception):

    def __init__(self, message: str, status_code: int | None = None, detail: dict | str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail if detail is not None else message


async def send_result_to_laravel(endpoint: str, payload: dict) -> dict:

    url = f"{settings.LARAVEL_API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    headers = {}

    if settings.LARAVEL_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.LARAVEL_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=settings.LARAVEL_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)

    except httpx.RequestError as exc:
        raise LaravelClientError(
            f"Could not reach Laravel API at {url}: {exc}",
            status_code=None,
        ) from exc

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        raise LaravelClientError(
            f"Laravel API returned {response.status_code}: {response.text}",
            status_code=response.status_code,
            detail=body,
        )

    try:
        return response.json()
    except ValueError:
        return {}