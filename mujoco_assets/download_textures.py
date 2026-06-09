"""Download and convert texture images for MuJoCo randomization demos."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request
from urllib.request import urlopen

from PIL import Image
from PIL import ImageOps

from mujoco_assets.textures import SOURCE_DIRECTORIES

USER_AGENT = "mujoco-assets-texture-fetcher/1.0"
DEFAULT_OUTPUT_ROOT = Path.cwd() / "textures"

GR00T_ISAACSIM_CFG_URL = (
  "https://raw.githubusercontent.com/NVlabs/GR00T-VisualSim2Real/"
  "92bf0863d4a9b6ee29849736152b7769bd45c49c/gr00t/rl/data/tasks/"
  "walk_stand_place_grasp_turn_homie/scenario_cfg/isaacsim.py"
)
ROBOSUITE_TEXTURES_API_URL = (
  "https://api.github.com/repos/ARISE-Initiative/robosuite/contents/robosuite/models/assets/textures?ref=master"
)
LIBERO_ASSETS_API_URL = "https://huggingface.co/api/datasets/lerobot/libero-assets"
LIBERO_RESOLVE_URL = "https://huggingface.co/datasets/lerobot/libero-assets/resolve/main"
POLYHAVEN_ASSETS_URL = "https://api.polyhaven.com/assets?t=textures"
POLYHAVEN_FILES_URL = "https://api.polyhaven.com/files"

SOURCE_INFO = {
  "robosuite": {
    "name": "robosuite",
    "directory": SOURCE_DIRECTORIES["robosuite"],
    "url": "https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/models/assets/textures",
    "license": "MIT",
  },
  "lerobot_libero_assets": {
    "name": "LeRobot LIBERO assets",
    "directory": SOURCE_DIRECTORIES["lerobot_libero_assets"],
    "url": "https://huggingface.co/datasets/lerobot/libero-assets",
    "license": "not declared by Hugging Face dataset metadata",
  },
  "nvidia_isaac_materials": {
    "name": "NVIDIA Isaac material textures discovered from GR00T-VisualSim2Real",
    "directory": SOURCE_DIRECTORIES["nvidia_isaac_materials"],
    "url": "https://github.com/NVlabs/GR00T-VisualSim2Real",
    "license": "NVIDIA Omniverse asset license",
  },
  "polyhaven": {
    "name": "Poly Haven textures",
    "directory": SOURCE_DIRECTORIES["polyhaven"],
    "url": "https://polyhaven.com/textures",
    "license": "CC0-1.0",
  },
}


@dataclasses.dataclass(frozen=True)
class TextureDownload:
  source: str
  source_path: str
  url: str
  label: str
  license: str


def _request(url: str) -> bytes:
  request = Request(url, headers={"User-Agent": USER_AGENT})
  with urlopen(request, timeout=45) as response:
    return response.read()


def _json(url: str):
  return json.loads(_request(url).decode("utf-8"))


def _text(url: str) -> str:
  return _request(url).decode("utf-8")


def _slug(text: str) -> str:
  slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
  return slug or "texture"


def _unique_filename(source: str, source_path: str, url: str) -> str:
  stem = _slug(Path(source_path).with_suffix("").as_posix())
  suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
  return f"{_slug(source)}_{stem}_{suffix}.png"


def _discover_robosuite() -> list[TextureDownload]:
  downloads = []
  for item in _json(ROBOSUITE_TEXTURES_API_URL):
    if item.get("type") != "file" or not item["name"].lower().endswith((".png", ".jpg", ".jpeg")):
      continue
    downloads.append(
      TextureDownload(
        source="robosuite",
        source_path=item["path"],
        url=item["download_url"],
        label=f"robosuite {Path(item['name']).stem.replace('-', ' ')}",
        license="MIT",
      )
    )
  return downloads


def _discover_libero() -> list[TextureDownload]:
  downloads = []
  for sibling in _json(LIBERO_ASSETS_API_URL)["siblings"]:
    path = sibling["rfilename"]
    if not path.lower().endswith((".png", ".jpg", ".jpeg")):
      continue
    downloads.append(
      TextureDownload(
        source="lerobot_libero_assets",
        source_path=path,
        url=f"{LIBERO_RESOLVE_URL}/{quote(path)}",
        label=f"LIBERO {Path(path).stem.replace('_', ' ')}",
        license="dataset license",
      )
    )
  return downloads


def _discover_nvidia_isaac() -> list[TextureDownload]:
  cfg = _text(GR00T_ISAACSIM_CFG_URL)
  mdl_urls = sorted(set(re.findall(r"https://[^\"']+?\.mdl", cfg)))
  downloads = []
  for mdl_url in mdl_urls:
    try:
      mdl = _text(mdl_url)
    except (HTTPError, URLError, TimeoutError):
      continue
    match = re.search(r"diffuse_texture:\s*texture_2d\(\"(?P<path>[^\"]+)\"", mdl)
    if not match:
      continue
    texture_path = match.group("path").replace("./", "")
    base_url = mdl_url.rsplit("/", 1)[0]
    material_name = Path(mdl_url).stem
    downloads.append(
      TextureDownload(
        source="nvidia_isaac_materials",
        source_path=texture_path,
        url=f"{base_url}/{texture_path}",
        label=f"NVIDIA Isaac {material_name.replace('_', ' ')}",
        license="NVIDIA Omniverse asset license",
      )
    )
  return downloads


def _polyhaven_diffuse_url(asset_id: str) -> str | None:
  files = _json(f"{POLYHAVEN_FILES_URL}/{asset_id}")
  diffuse = files.get("Diffuse") or files.get("diffuse") or files.get("Albedo") or files.get("Color")
  if not diffuse:
    return None
  for size in ("1k", "2k", "4k"):
    by_size = diffuse.get(size)
    if not by_size:
      continue
    for extension in ("jpg", "png"):
      if extension in by_size:
        return by_size[extension]["url"]
  return None


def _discover_polyhaven(limit: int | None) -> list[TextureDownload]:
  assets = _json(POLYHAVEN_ASSETS_URL)
  asset_ids = sorted(assets)
  if limit is not None:
    asset_ids = asset_ids[:limit]

  downloads = []
  with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(_polyhaven_diffuse_url, asset_id): asset_id for asset_id in asset_ids}
    for future in as_completed(futures):
      asset_id = futures[future]
      try:
        url = future.result()
      except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        continue
      if not url:
        continue
      meta = assets[asset_id]
      downloads.append(
        TextureDownload(
          source="polyhaven",
          source_path=asset_id,
          url=url,
          label=f"Poly Haven {meta.get('name', asset_id)}",
          license="CC0-1.0",
        )
      )
  return sorted(downloads, key=lambda item: item.source_path)


def _target_dir(output_root: Path, source: str) -> Path:
  return output_root / SOURCE_DIRECTORIES[source]


def _convert(download: TextureDownload, output_root: Path, size: int) -> dict | None:
  try:
    data = _request(download.url)
  except (HTTPError, URLError, TimeoutError):
    return None

  source_sha256 = hashlib.sha256(data).hexdigest()
  output_dir = _target_dir(output_root, download.source)
  output_dir.mkdir(parents=True, exist_ok=True)

  try:
    with Image.open(io.BytesIO(data)) as image:
      image = ImageOps.exif_transpose(image).convert("RGB")
      image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
      filename = _unique_filename(download.source, download.source_path, download.url)
      tmp_path = Path(tempfile.mkstemp(suffix=".png", dir=output_dir)[1])
      try:
        image.save(tmp_path, format="PNG", optimize=True)
        output_path = output_dir / filename
        tmp_path.replace(output_path)
      finally:
        if tmp_path.exists():
          tmp_path.unlink()
  except Exception:
    return None

  relative_file = f"{SOURCE_DIRECTORIES[download.source]}/{filename}"
  output_sha256 = hashlib.sha256((output_root / relative_file).read_bytes()).hexdigest()
  return {
    "name": f"texture_{_slug(download.source)}_{_slug(download.source_path)}_{source_sha256[:8]}",
    "file": relative_file,
    "label": download.label,
    "source": download.source,
    "source_path": download.source_path,
    "source_url": download.url,
    "license": download.license,
    "source_sha256": source_sha256,
    "sha256": output_sha256,
  }


def _dedupe(downloads: Sequence[TextureDownload]) -> list[TextureDownload]:
  by_url = {}
  for download in downloads:
    by_url.setdefault(download.url, download)
  return sorted(by_url.values(), key=lambda item: (item.source, item.source_path, item.url))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  parser.add_argument("--size", type=int, default=256, help="converted square texture size in pixels")
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--skip-polyhaven", action="store_true")
  parser.add_argument(
    "--polyhaven-limit",
    type=int,
    default=0,
    help="limit Poly Haven textures; 0 downloads every diffuse texture found",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.size < 1:
    raise ValueError(f"--size must be positive, got {args.size}.")
  if args.workers < 1:
    raise ValueError(f"--workers must be positive, got {args.workers}.")

  output_root = args.output_root.expanduser().resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  for directory in SOURCE_DIRECTORIES.values():
    (output_root / directory).mkdir(parents=True, exist_ok=True)

  downloads = []
  downloads.extend(_discover_robosuite())
  downloads.extend(_discover_libero())
  downloads.extend(_discover_nvidia_isaac())
  if not args.skip_polyhaven:
    polyhaven_limit = None if args.polyhaven_limit == 0 else args.polyhaven_limit
    downloads.extend(_discover_polyhaven(polyhaven_limit))
  downloads = _dedupe(downloads)

  converted = []
  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(_convert, download, output_root, args.size) for download in downloads]
    for index, future in enumerate(as_completed(futures), start=1):
      entry = future.result()
      if entry is not None:
        converted.append(entry)
      if index % 50 == 0 or index == len(futures):
        print(f"converted {len(converted)}/{index} successful textures")

  converted.sort(key=lambda item: (item["source"], item["source_path"], item["file"]))
  counts = {source: sum(item["source"] == source for item in converted) for source in SOURCE_DIRECTORIES}
  sources = []
  for source, info in SOURCE_INFO.items():
    source_info = dict(info)
    source_info["id"] = source
    source_info["count"] = counts[source]
    sources.append(source_info)

  manifest = {
    "description": "Converted texture dataset for MuJoCo texture randomization demos.",
    "format": {
      "file_type": "png",
      "color_mode": "RGB",
      "width": args.size,
      "height": args.size,
      "conversion": "Source images were center-cropped and resampled to square RGB PNG files.",
    },
    "source_directories": SOURCE_DIRECTORIES,
    "sources": sources,
    "textures": converted,
  }
  manifest_path = output_root / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"wrote {len(converted)} textures to {output_root}")
  print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
  main()
