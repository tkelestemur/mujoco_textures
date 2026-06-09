"""Helpers for loading and sampling packaged MuJoCo texture assets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Iterable
from typing import Sequence

SOURCE_DIRECTORIES = {
  "robosuite": "robosuites",
  "lerobot_libero_assets": "lerobot-libero",
  "nvidia_isaac_materials": "isaac-lib",
  "polyhaven": "polyhaven",
}


@dataclasses.dataclass(frozen=True)
class TextureAsset:
  """A converted texture image plus source metadata from the manifest."""

  name: str
  file: Path
  label: str
  source: str
  source_path: str
  source_url: str
  license: str
  sha256: str
  source_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class TextureManifest:
  """Texture manifest loaded from a checkout or installed package."""

  path: Path
  texture_root: Path
  textures: tuple[TextureAsset, ...]
  sources: tuple[dict, ...]
  format: dict

  def by_source(self, source: str) -> tuple[TextureAsset, ...]:
    return tuple(texture for texture in self.textures if texture.source == source)

  def sample(
    self,
    count: int,
    *,
    source: str | None = None,
    seed: int | None = None,
    replace: bool = False,
  ) -> tuple[TextureAsset, ...]:
    textures = self.by_source(source) if source is not None else self.textures
    return _sample(textures, count, seed=seed, replace=replace)


def _package_dir() -> Path:
  return Path(__file__).resolve().parent


def default_texture_root() -> Path:
  """Return the packaged texture root for either source checkouts or wheels."""

  candidates = (
    _package_dir().parent / "textures",
    _package_dir() / "textures",
  )
  for candidate in candidates:
    if (candidate / "manifest.json").exists():
      return candidate
  return candidates[0]


def load_manifest(texture_root: str | Path | None = None, manifest_path: str | Path | None = None) -> TextureManifest:
  """Load texture metadata and resolve each image path."""

  if manifest_path is not None:
    path = Path(manifest_path).expanduser().resolve()
    root = path.parent
  else:
    root = Path(texture_root).expanduser().resolve() if texture_root is not None else default_texture_root()
    path = root / "manifest.json"

  if not path.exists():
    raise FileNotFoundError(f"texture manifest not found: {path}")

  payload = json.loads(path.read_text())
  textures = []
  for entry in payload.get("textures", []):
    file_path = root / entry["file"]
    textures.append(
      TextureAsset(
        name=entry["name"],
        file=file_path,
        label=entry.get("label", entry["name"]),
        source=entry["source"],
        source_path=entry.get("source_path", ""),
        source_url=entry.get("source_url", ""),
        license=entry.get("license", ""),
        sha256=entry.get("sha256", ""),
        source_sha256=entry.get("source_sha256"),
      )
    )

  return TextureManifest(
    path=path,
    texture_root=root,
    textures=tuple(textures),
    sources=tuple(payload.get("sources", ())),
    format=payload.get("format", {}),
  )


def list_textures(source: str | None = None, texture_root: str | Path | None = None) -> tuple[TextureAsset, ...]:
  """Return all textures, optionally filtered by source id."""

  manifest = load_manifest(texture_root)
  return manifest.by_source(source) if source is not None else manifest.textures


def sample_textures(
  count: int,
  *,
  source: str | None = None,
  seed: int | None = None,
  replace: bool = False,
  texture_root: str | Path | None = None,
) -> tuple[TextureAsset, ...]:
  """Sample texture assets from the manifest."""

  manifest = load_manifest(texture_root)
  return manifest.sample(count, source=source, seed=seed, replace=replace)


def _sample(textures: Sequence[TextureAsset], count: int, *, seed: int | None, replace: bool) -> tuple[TextureAsset, ...]:
  if count < 1:
    raise ValueError(f"count must be positive, got {count}")
  if not textures:
    raise ValueError("cannot sample from an empty texture set")
  rng = random.Random(seed)
  if replace:
    return tuple(rng.choice(textures) for _ in range(count))
  if count > len(textures):
    raise ValueError(f"cannot sample {count} unique textures from {len(textures)} available textures")
  return tuple(rng.sample(tuple(textures), count))


def _source_counts(textures: Iterable[TextureAsset]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for texture in textures:
    counts[texture.source] = counts.get(texture.source, 0) + 1
  return counts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--texture-root", type=Path, default=None, help="directory containing manifest.json")
  parser.add_argument("--source", choices=sorted(SOURCE_DIRECTORIES), default=None)
  parser.add_argument("--limit", type=int, default=20)
  parser.add_argument("--paths", action="store_true", help="print resolved image paths")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  manifest = load_manifest(args.texture_root)
  textures = manifest.by_source(args.source) if args.source is not None else manifest.textures
  counts = _source_counts(textures)
  print(f"manifest: {manifest.path}")
  print(f"textures: {len(textures)}")
  for source, count in sorted(counts.items()):
    print(f"  {source}: {count}")
  for texture in textures[: max(args.limit, 0)]:
    value = texture.file if args.paths else texture.name
    print(value)


if __name__ == "__main__":
  main()
