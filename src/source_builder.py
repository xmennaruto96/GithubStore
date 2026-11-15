import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.settings import RepositoryConfig, Settings


class GithubReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class GithubRepo:
    owner: str
    name: str


def _repo_from_url(value: str) -> GithubRepo:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower()
    if host != "github.com":
        raise GithubReleaseError(f"Only github.com repository URLs are supported: {value}")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise GithubReleaseError(f"Invalid GitHub repository URL: {value}")
    return GithubRepo(owner=parts[0], name=parts[1].removesuffix(".git"))


def _request_json(url: str, settings: Settings) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "GithubStore",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            data = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GithubReleaseError(f"GitHub API error {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise GithubReleaseError(f"Could not reach GitHub API for {url}: {exc.reason}") from exc

    value = json.loads(data)
    if not isinstance(value, (dict, list)):
        raise GithubReleaseError(f"Unexpected GitHub API response for {url}")
    return value


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


_EPOCH = datetime.min.replace(tzinfo=UTC)


def _release_timestamp(release: dict, assets: list[dict]) -> datetime:
    """When this build really appeared, as opposed to what the release claims.

    - `created_at` on a release is the date of the *commit* it points at, so a
      fresh build of an old commit looks ancient. It is only a last resort.
    - Nightly / rolling releases keep one tag and one `published_at` forever and
      just re-upload the asset, so the asset's own created/updated time is the
      only honest signal.
    The newest of all of these wins.
    """
    stamps = [_parse_ts(release.get("published_at"))]
    for asset in assets:
        stamps.append(_parse_ts(asset.get("created_at")))
        stamps.append(_parse_ts(asset.get("updated_at")))
    found = [s for s in stamps if s is not None]
    if found:
        return max(found)
    return _parse_ts(release.get("created_at")) or _EPOCH


def _select_release(releases: object, include_prereleases: bool) -> dict | None:
    """Newest installable release by real timestamp, not by list order."""
    if not isinstance(releases, list):
        return None
    best: tuple[datetime, dict] | None = None
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        if release.get("prerelease") and not include_prereleases:
            continue
        assets = _ipa_assets(release)
        if not assets:
            continue
        stamp = _release_timestamp(release, assets)
        if best is None or stamp > best[0]:
            best = (stamp, release)
    return best[1] if best else None


def _latest_release(settings: Settings, repo: GithubRepo, include_prereleases: bool) -> dict:
    releases = _request_json(
        f"https://api.github.com/repos/{repo.owner}/{repo.name}/releases"
        f"?per_page={settings.github_max_releases}",
        settings,
    )
    release = _select_release(releases, include_prereleases)
    if release is None:
        kind = "release" if include_prereleases else "stable release"
        raise GithubReleaseError(f"No {kind} with an .ipa asset found for {repo.owner}/{repo.name}")
    return release


def _ipa_assets(release: dict) -> list[dict]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        return []
    return [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and str(asset.get("name") or "").lower().endswith(".ipa")
        and str(asset.get("browser_download_url") or "")
    ]


def _asset_sort_key(asset: dict) -> tuple[int, int]:
    name = str(asset.get("name") or "").lower()
    penalty = 1 if any(part in name for part in ("debug", "symbols", "dsyms")) else 0
    size = int(asset.get("size") or 0)
    return penalty, -size


def _release_ipa_assets(release: dict, repo: GithubRepo) -> list[dict]:
    assets = sorted(_ipa_assets(release), key=_asset_sort_key)
    if not assets:
        raise GithubReleaseError(f"No .ipa asset found on the selected release for {repo.owner}/{repo.name}")
    return assets


def _clean_name(value: str) -> str:
    name = Path(value).name
    if name.lower().endswith(".ipa"):
        name = name[:-4]
    name = re.sub(r"[_\-.]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "GitHub App"


def _identifier_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".")
    return component or "app"


_DOTTED_VERSION = re.compile(r"\d+(?:\.\d+)+[0-9A-Za-z.+_-]*")


def _display_version(release: dict) -> str:
    """Best human version: the tag, unless the tag is a bare build number.

    Some projects tag builds as "30" and keep the real version ("3.0.13-30") in
    the release title or the IPA filename. In that case the first dotted
    version found in the title, then in the asset names, wins.
    """
    tag = str(release.get("tag_name") or "").strip()
    name = str(release.get("name") or "").strip()
    if "." not in tag:
        sources = [name] + [
            Path(str(a.get("name") or "")).stem
            for a in sorted(_ipa_assets(release), key=_asset_sort_key)
        ]
        for text in sources:
            match = _DOTTED_VERSION.search(text)
            if match:
                return match.group(0).rstrip(".-_+")
    return tag or name


def _release_version(release: dict) -> str:
    version = _display_version(release).lstrip("v") or "1.0"
    # Rolling tags ("nightly", "latest", "continuous") never change, so a client
    # would never see an update. Make them unique per build.
    if not any(ch.isdigit() for ch in version):
        version = f"{version}-{_release_stamp(release):%Y%m%d.%H%M}"
    return version


def _release_stamp(release: dict) -> datetime:
    return _release_timestamp(release, _ipa_assets(release))


def _build_version(release: dict, version: str) -> str:
    # Pre-releases are often re-uploaded under the same tag; the timestamp keeps
    # each upload distinguishable.
    if release.get("prerelease"):
        stamp = f"{_release_stamp(release):%Y%m%d%H%M}"
        return version if stamp in version.replace(".", "").replace("-", "") else f"{version}+{stamp}"
    return version


def _release_date(release: dict) -> str:
    stamp = _release_stamp(release)
    if stamp > _EPOCH:
        return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _description(release: dict, repo: GithubRepo) -> str:
    body = str(release.get("body") or "").strip()
    name = str(release.get("name") or release.get("tag_name") or "").strip()
    text = body or name or f"Latest GitHub release for {repo.owner}/{repo.name}."
    if release.get("prerelease"):
        return f"Pre-release\n\n{text}"
    return text


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def _repo_icon_url(settings: Settings, repo_config: RepositoryConfig) -> str:
    if repo_config.icon:
        if _is_http_url(repo_config.icon):
            return repo_config.icon
        return f"{settings.base_url}/icon/{repo_config.slug}.png"
    return settings.source_icon_url or f"{settings.base_url}/source-icon.png"


def _app_from_release(
    settings: Settings,
    repo_config: RepositoryConfig,
    repo: GithubRepo,
    release: dict,
    asset: dict,
    include_asset_identity: bool,
) -> dict:
    filename = str(asset.get("name") or f"{repo.name}.ipa")
    version = _release_version(release)
    description = _description(release, repo)
    download_url = str(asset["browser_download_url"])
    name = _clean_name(filename) if include_asset_identity else repo_config.name
    bundle_identifier = f"github.{repo.owner.lower()}.{repo.name.lower()}"
    if include_asset_identity:
        bundle_identifier = f"{bundle_identifier}.{_identifier_component(Path(filename).stem)}"

    return {
        "name": name,
        "bundleIdentifier": bundle_identifier,
        "developerName": repo.owner,
        "subtitle": f"{repo.owner}/{repo.name}",
        "localizedDescription": description,
        "iconURL": _repo_icon_url(settings, repo_config),
        "tintColor": repo_config.tint_color,
        "versions": [
            {
                "version": version,
                "buildVersion": _build_version(release, version),
                "date": _release_date(release),
                "downloadURL": download_url,
                "localizedDescription": description,
                "size": int(asset.get("size") or 0),
            }
        ],
    }


def _build_apps(settings: Settings, repo_config: RepositoryConfig) -> list[dict]:
    repo = _repo_from_url(repo_config.url)
    release = _latest_release(settings, repo, repo_config.prereleases)
    assets = _release_ipa_assets(release, repo)
    include_asset_identity = len(assets) > 1
    return [
        _app_from_release(settings, repo_config, repo, release, asset, include_asset_identity)
        for asset in assets
    ]


async def build_source(settings: Settings) -> dict:
    source: dict = {
        "name": settings.source_name,
        "subtitle": settings.source_subtitle,
        "description": settings.source_description,
        "iconURL": settings.source_icon_url or f"{settings.base_url}/source-icon.png",
        "tintColor": settings.source_tint_color,
        "apps": [],
    }
    errors: list[dict[str, str]] = []

    results = await asyncio.gather(
        *[asyncio.to_thread(_build_apps, settings, repo) for repo in settings.repositories],
        return_exceptions=True,
    )
    for repo_config, result in zip(settings.repositories, results, strict=True):
        if isinstance(result, Exception):
            errors.append(
                {
                    "name": repo_config.name,
                    "url": repo_config.url,
                    "error": str(result),
                }
            )
            continue
        source["apps"].extend(result)

    if errors:
        source["errors"] = errors
    return source
