"""Visualize per-world material texture randomization with MuJoCo Warp and Viser."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

import numpy as np

from mujoco_textures.textures import SOURCE_DIRECTORIES
from mujoco_textures.textures import TextureAsset
from mujoco_textures.textures import default_texture_root
from mujoco_textures.textures import load_manifest

mujoco = None
wp = None
mjw = None

TABLE_HALF_EXTENTS = (0.60, 0.45)
TABLE_TOP_HALF_Z = 0.035
TABLE_TOP_CENTER_Z = -TABLE_TOP_HALF_Z
TABLE_LEG_HALF_Z = 0.20
TABLE_LEG_CENTER_Z = TABLE_TOP_CENTER_Z - TABLE_TOP_HALF_Z - TABLE_LEG_HALF_Z

FR3_CTRL_BASE = np.array([0.0, -0.55, 0.0, -1.85, 0.0, 1.95, -0.7853], dtype=np.float32)
FR3_CTRL_AMP = np.array([0.45, 0.22, 0.38, 0.28, 0.35, 0.30, 0.55], dtype=np.float32)
FR3_CTRL_PHASES = np.array([0.0, 0.7, 1.6, 2.4, 3.1, 3.8, 4.5], dtype=np.float32)
FR3_INITIAL_QPOS = {
  "fr3v2_joint1": 0.0,
  "fr3v2_joint2": -0.55,
  "fr3v2_joint3": 0.0,
  "fr3v2_joint4": -1.85,
  "fr3v2_joint5": 0.0,
  "fr3v2_joint6": 1.95,
  "fr3v2_joint7": -0.7853,
}
MENAGERIE_FR3_API_ROOT = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/franka_fr3_v2"
MENAGERIE_REF = "main"
USER_AGENT = "mujoco-textures-fr3-fetcher/1.0"


@dataclasses.dataclass(frozen=True)
class TextureSpec:
  name: str
  file: Path
  label: str


def _ensure_visual_deps() -> None:
  global mjw, mujoco, wp

  if mujoco is not None and wp is not None and mjw is not None:
    return

  try:
    import mujoco as mujoco_module
    import mujoco_warp as mjw_module
    import warp as wp_module
  except ImportError as exc:
    raise RuntimeError(
      "This demo requires visualization dependencies. Run with `uv run --extra visualize "
      "scripts/randomize_textures_viser`, or install `mujoco-textures[visualize]`."
    ) from exc

  mjw = mjw_module
  mujoco = mujoco_module
  wp = wp_module


def _cache_root() -> Path:
  xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
  if xdg_cache_home:
    return Path(xdg_cache_home).expanduser()
  return Path.home() / ".cache"


def _default_fr3_dir() -> Path:
  return _cache_root() / "mujoco_textures" / "mujoco_menagerie" / "franka_fr3_v2"


def _default_fr3_candidates() -> tuple[Path, ...]:
  return (
    Path.cwd() / "models/franka_fr3_v2/fr3v2.xml",
    Path.cwd() / "franka_fr3_v2/fr3v2.xml",
    Path.home() / "code/mujoco_menagerie/franka_fr3_v2/fr3v2.xml",
    Path.home() / "code/mujoco_warp/benchmarks/mujoco_menagerie/franka_fr3_v2/fr3v2.xml",
    _default_fr3_dir() / "fr3v2.xml",
  )


def _request(url: str) -> bytes:
  request = Request(url, headers={"User-Agent": USER_AGENT})
  with urlopen(request, timeout=60) as response:
    return response.read()


def _json(url: str):
  return json.loads(_request(url).decode("utf-8"))


def _with_ref(api_url: str) -> str:
  if "ref=" in api_url:
    return api_url
  separator = "&" if "?" in api_url else "?"
  return f"{api_url}{separator}ref={MENAGERIE_REF}"


def _download_menagerie_dir(api_url: str, target_dir: Path) -> None:
  target_dir.mkdir(parents=True, exist_ok=True)
  for item in _json(_with_ref(api_url)):
    path = target_dir / item["name"]
    if item["type"] == "dir":
      _download_menagerie_dir(item["url"], path)
    elif item["type"] == "file":
      if path.exists() and path.stat().st_size == int(item["size"]):
        continue
      tmp_path = path.with_suffix(path.suffix + ".tmp")
      tmp_path.write_bytes(_request(item["download_url"]))
      tmp_path.replace(path)


def _ensure_fr3_model(fr3_dir: Path) -> Path:
  fr3_xml = fr3_dir / "fr3v2.xml"
  if fr3_xml.exists():
    return fr3_xml.resolve()

  print(f"Downloading Menagerie franka_fr3_v2 model to {fr3_dir}")
  try:
    _download_menagerie_dir(MENAGERIE_FR3_API_ROOT, fr3_dir)
  except (HTTPError, URLError, TimeoutError) as exc:
    raise RuntimeError(f"failed to download Menagerie franka_fr3_v2 model to {fr3_dir}") from exc

  if not fr3_xml.exists():
    raise FileNotFoundError(f"downloaded Menagerie franka_fr3_v2 is missing {fr3_xml}")
  return fr3_xml.resolve()


def _resolve_fr3_xml(fr3_xml: Path | None, fr3_dir: Path | None, download_fr3: bool) -> Path:
  if fr3_xml is not None:
    path = fr3_xml.expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"FR3 XML not found: {path}")
    return path

  if fr3_dir is not None:
    candidate = fr3_dir.expanduser().resolve() / "fr3v2.xml"
    if candidate.exists():
      return candidate
    if download_fr3:
      return _ensure_fr3_model(fr3_dir.expanduser().resolve())

  for candidate in _default_fr3_candidates():
    if candidate.exists():
      return candidate.resolve()

  if download_fr3:
    return _ensure_fr3_model(_default_fr3_dir())

  raise FileNotFoundError("FR3 XML not found. Pass --fr3-xml, --fr3-dir, or allow the default Menagerie download.")


def _resolve_meshdir(model_xml: Path, meshdir: str) -> Path:
  path = Path(meshdir) if meshdir else model_xml.parent
  path = path if path.is_absolute() else model_xml.parent / path
  return path.resolve()


def _select_texture_specs(
  assets: Sequence[TextureAsset],
  *,
  source: str | None,
  max_textures: int,
  seed: int,
) -> tuple[TextureSpec, ...]:
  filtered = [asset for asset in assets if source is None or asset.source == source]
  if not filtered:
    raise ValueError(f"no textures found for source {source!r}")
  if max_textures > 0 and len(filtered) > max_textures:
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(filtered), size=max_textures, replace=False))
    filtered = [filtered[int(index)] for index in indices]
  specs = []
  seen_names = set()
  for asset in filtered:
    if asset.name in seen_names:
      raise ValueError(f"duplicate texture name in manifest: {asset.name!r}")
    if not asset.file.exists():
      raise FileNotFoundError(f"texture file listed in manifest does not exist: {asset.file}")
    seen_names.add(asset.name)
    specs.append(TextureSpec(name=asset.name, file=asset.file.resolve(), label=asset.label))
  return tuple(specs)


def _add_demo_assets(spec: mujoco.MjSpec, texture_specs: Sequence[TextureSpec]) -> None:
  for tex in texture_specs:
    spec.add_texture(
      name=tex.name,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      file=tex.file.as_posix(),
    )

  spec.add_material(
    name="table_mat",
    textures=["", texture_specs[0].name],
    texrepeat=[7, 5],
    texuniform=False,
    rgba=[1, 1, 1, 1],
  )
  spec.add_material(name="table_leg_mat", rgba=[0.25, 0.27, 0.29, 1])
  spec.add_material(name="floor_mat", rgba=[0.42, 0.43, 0.44, 1])
  spec.add_material(name="block_mat", rgba=[0.8, 0.82, 0.84, 1])


def _add_demo_worldbody(spec: mujoco.MjSpec, include_standalone_blocks: bool) -> None:
  worldbody = spec.worldbody
  worldbody.add_camera(name="demo", pos=[0, -1.65, 0.95], xyaxes=[1, 0, 0, 0, 0.54, 0.84], fovy=46)
  worldbody.add_light(name="demo_key", pos=[-0.5, -1.0, 2.0], dir=[0.3, 0.5, -1], diffuse=[0.9, 0.9, 0.85])
  worldbody.add_light(name="demo_fill", pos=[1.2, 0.8, 1.2], dir=[-0.6, -0.3, -1], diffuse=[0.35, 0.40, 0.45])
  worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=[0, 0, -0.44],
    size=[3, 3, 0.02],
    material="floor_mat",
    contype=0,
    conaffinity=0,
  )

  table = worldbody.add_body(name="table", mocap=True)
  table.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0, 0, TABLE_TOP_CENTER_Z],
    size=[TABLE_HALF_EXTENTS[0], TABLE_HALF_EXTENTS[1], TABLE_TOP_HALF_Z],
    material="table_mat",
    group=1,
    contype=0,
    conaffinity=0,
  )
  for name, x, y in (
    ("leg_fl", 0.48, 0.34),
    ("leg_fr", 0.48, -0.34),
    ("leg_bl", -0.48, 0.34),
    ("leg_br", -0.48, -0.34),
  ):
    table.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[x, y, TABLE_LEG_CENTER_Z],
      size=[0.035, 0.035, TABLE_LEG_HALF_Z],
      material="table_leg_mat",
      contype=0,
      conaffinity=0,
    )

  if include_standalone_blocks:
    for index, x in enumerate((-0.28, 0.0, 0.28)):
      body = worldbody.add_body(name=f"demo_block_{index}", pos=[x, 0.05 * index, 0.08])
      body.add_freejoint()
      body.add_geom(
        name=f"demo_block_geom_{index}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.055, 0.055, 0.055],
        material="block_mat",
      )


def _build_demo_spec(fr3_xml: Path, texture_specs: Sequence[TextureSpec]) -> mujoco.MjSpec:
  _ensure_visual_deps()

  spec = mujoco.MjSpec()
  spec.modelname = "mujoco_textures_randomization"
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.visual.headlight.active = 0
  spec.visual.quality.shadowsize = 2048

  fr3 = mujoco.MjSpec.from_file(fr3_xml.as_posix())
  spec.meshdir = _resolve_meshdir(fr3_xml, fr3.meshdir).as_posix()
  fr3.meshdir = spec.meshdir
  fr3_anchor = spec.worldbody.add_body(name="fr3_anchor", mocap=True)
  fr3_frame = fr3_anchor.add_frame(name="fr3_mount")
  spec.attach(fr3, frame=fr3_frame, prefix="")

  _add_demo_assets(spec, texture_specs)
  _add_demo_worldbody(spec, include_standalone_blocks=False)
  return spec


def _set_initial_state(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> None:
  _ensure_visual_deps()

  for joint_name, value in FR3_INITIAL_QPOS.items():
    joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id >= 0:
      mjd.qpos[mjm.jnt_qposadr[joint_id]] = value

  if mjm.nu:
    mjd.ctrl[: min(mjm.nu, FR3_CTRL_BASE.size)] = FR3_CTRL_BASE[: min(mjm.nu, FR3_CTRL_BASE.size)]
  mjm.qpos0[:] = mjd.qpos
  mujoco.mj_forward(mjm, mjd)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=4, help="number of parallel worlds")
  parser.add_argument("--randomize-every", type=int, default=50, help="texture reset interval in physics steps")
  parser.add_argument("--width", type=int, default=192, help="Warp render width per environment")
  parser.add_argument("--height", type=int, default=144, help="Warp render height per environment")
  parser.add_argument("--render-every", type=int, default=1, help="render every N Viser frames")
  parser.add_argument("--seed", type=int, default=0, help="random seed for texture choices")
  parser.add_argument("--device", default=None, help="Warp device, for example cuda:0 or cpu")
  parser.add_argument("--port", type=int, default=8080, help="Viser server port")
  parser.add_argument("--fr3-xml", type=Path, default=None, help="optional path to Menagerie franka_fr3_v2/fr3v2.xml")
  parser.add_argument("--fr3-dir", type=Path, default=None, help="optional directory containing franka_fr3_v2 assets")
  parser.add_argument("--no-download-fr3", action="store_true", help="do not download Menagerie FR3 assets if missing")
  parser.add_argument("--texture-root", type=Path, default=default_texture_root(), help="directory containing manifest.json")
  parser.add_argument("--source", choices=sorted(SOURCE_DIRECTORIES), default=None, help="sample from one texture source")
  parser.add_argument("--max-textures", type=int, default=64, help="maximum textures to compile; 0 uses every manifest entry")
  return parser.parse_args(argv)


def _env_offsets(num_envs: int) -> np.ndarray:
  cols = int(math.ceil(math.sqrt(num_envs)))
  rows = int(math.ceil(num_envs / cols))
  spacing_x = 1.75
  spacing_y = 1.55
  offsets = np.zeros((num_envs, 3), dtype=np.float32)
  for world in range(num_envs):
    row, col = divmod(world, cols)
    offsets[world, 0] = (col - (cols - 1) * 0.5) * spacing_x
    offsets[world, 1] = ((rows - 1) * 0.5 - row) * spacing_y
  return offsets


def _make_texture_preview(mjm: mujoco.MjModel, texture_id: int) -> np.ndarray:
  _ensure_visual_deps()

  width = int(mjm.tex_width[texture_id])
  height = int(mjm.tex_height[texture_id])
  nchannel = int(mjm.tex_nchannel[texture_id])
  adr = int(mjm.tex_adr[texture_id])
  if width < 1 or height < 1 or nchannel < 1 or adr < 0:
    raise ValueError(f"texture id {texture_id} has invalid compiled texture data")

  image = mjm.tex_data[adr : adr + width * height * nchannel].reshape(height, width, nchannel)
  if nchannel == 1:
    return np.repeat(image, 3, axis=2).astype(np.uint8)
  return image[:, :, :3].copy().astype(np.uint8)


def _unpack_rgb(packed_row: np.ndarray, width: int, height: int) -> np.ndarray:
  packed = packed_row.reshape(height, width).astype(np.uint32)
  b = (packed & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  return np.dstack([r, g, b])


def _patch_mjviser_compat() -> None:
  _ensure_visual_deps()

  if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    setattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", 0)


def _control_targets(step: int, num_envs: int, ctrlrange: np.ndarray) -> np.ndarray:
  nu = ctrlrange.shape[0]
  if nu == 0:
    return np.zeros((num_envs, 0), dtype=np.float32)
  base = FR3_CTRL_BASE[:nu]
  amp = FR3_CTRL_AMP[:nu]
  phases = FR3_CTRL_PHASES[:nu]
  targets = np.empty((num_envs, nu), dtype=np.float32)
  for world in range(num_envs):
    phase = step * 0.025 + world * 0.55
    targets[world] = base + amp * np.sin(phase + phases)
  return np.clip(targets, ctrlrange[:, 0], ctrlrange[:, 1]).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.num_envs < 1:
    raise ValueError(f"--num-envs must be positive, got {args.num_envs}.")
  if args.randomize_every < 1:
    raise ValueError(f"--randomize-every must be positive, got {args.randomize_every}.")
  if args.render_every < 1:
    raise ValueError(f"--render-every must be positive, got {args.render_every}.")

  _ensure_visual_deps()
  try:
    import viser
    from mjviser import Viewer as MjViserViewer
  except ImportError as exc:
    raise RuntimeError("This demo requires Viser dependencies. Run with `uv run --extra visualize`.") from exc
  _patch_mjviser_compat()

  manifest = load_manifest(args.texture_root)
  texture_specs = _select_texture_specs(
    manifest.textures,
    source=args.source,
    max_textures=args.max_textures,
    seed=args.seed,
  )
  fr3_xml = _resolve_fr3_xml(args.fr3_xml, args.fr3_dir, download_fr3=not args.no_download_fr3)

  wp.config.quiet = True
  wp.init()
  if args.device is not None:
    wp.set_device(args.device)

  mjm = _build_demo_spec(fr3_xml, texture_specs).compile()
  mjd = mujoco.MjData(mjm)
  _set_initial_state(mjm, mjd)

  num_envs = int(args.num_envs)
  rng = np.random.default_rng(args.seed)
  env_offsets = _env_offsets(num_envs)
  ctrlrange = mjm.actuator_ctrlrange.astype(np.float32)

  with wp.ScopedDevice(args.device):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=num_envs)
    mjw.forward(m, d)
    if not hasattr(mjw, "expand_mat_texid"):
      raise RuntimeError("This demo requires mujoco_warp.expand_mat_texid for per-world texture IDs.")
    mjw.expand_mat_texid(m, num_envs)

    rc = mjw.create_render_context(
      mjm,
      nworld=num_envs,
      cam_res=(args.width, args.height),
      render_rgb=True,
      render_depth=False,
      render_seg=False,
      use_textures=True,
      use_shadows=True,
      enabled_geom_groups=[0, 1, 2],
      enable_specular=True,
      enable_emission=False,
    )

    camera_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "demo")
    table_top_geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_mat_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
    texture_ids = np.array(
      [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_TEXTURE, spec.name) for spec in texture_specs], dtype=np.int32
    )
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    mat_texid = m.mat_texid.numpy()
    selected_texture = np.zeros(num_envs, dtype=np.int32)
    texture_previews = tuple(_make_texture_preview(mjm, int(texture_id)) for texture_id in texture_ids)

    table_image_handles: list[viser.ImageHandle] = []
    render_image_handles: list[viser.ImageHandle] = []
    label_handles: list[viser.LabelHandle] = []
    step_count = 0
    render_count = 0

    def set_table_textures() -> None:
      nonlocal selected_texture
      if len(texture_ids) >= num_envs:
        selected_texture = rng.choice(len(texture_ids), size=num_envs, replace=False).astype(np.int32)
      else:
        selected_texture = rng.integers(0, len(texture_ids), size=num_envs, dtype=np.int32)

      mat_texid[:, table_mat_id, rgb_role] = texture_ids[selected_texture]
      m.mat_texid.assign(mat_texid)

      for world, handle in enumerate(table_image_handles):
        handle.image = texture_previews[int(selected_texture[world])]
      for world, handle in enumerate(label_handles):
        handle.text = f"env {world} | {texture_specs[int(selected_texture[world])].label}"

    def _scene_offset(scene) -> np.ndarray:
      return np.asarray(getattr(scene, "_scene_offset", np.zeros(3)), dtype=np.float32)

    def table_surface_positions(scene) -> np.ndarray:
      positions = d.geom_xpos.numpy()[:, table_top_geom_id, :] + env_offsets + _scene_offset(scene)
      positions[:, 2] += TABLE_TOP_HALF_Z + 0.006
      return positions.astype(np.float32)

    def update_custom_visual_positions(scene) -> None:
      scene_offset = _scene_offset(scene)
      positions = table_surface_positions(scene)
      for world, handle in enumerate(table_image_handles):
        handle.position = positions[world]
      for world, handle in enumerate(render_image_handles):
        handle.position = env_offsets[world] + scene_offset + np.array([0.0, -1.00, 0.72], dtype=np.float32)
      for world, handle in enumerate(label_handles):
        handle.position = env_offsets[world] + scene_offset + np.array([-0.55, -0.78, 0.52], dtype=np.float32)

    def ensure_viser_handles(scene) -> None:
      if table_image_handles:
        return

      blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
      scene_offset = _scene_offset(scene)
      table_positions = table_surface_positions(scene)
      table_w = TABLE_HALF_EXTENTS[0] * 2.0
      table_h = TABLE_HALF_EXTENTS[1] * 2.0
      panel_w = 0.62
      panel_h = panel_w * args.height / args.width
      panel_wxyz = (0.70710678, 0.70710678, 0.0, 0.0)

      with scene.server.atomic():
        for world, offset in enumerate(env_offsets):
          table_image_handles.append(
            scene.server.scene.add_image(
              f"/texture_demo/env_{world}/table_texture",
              texture_previews[int(selected_texture[world])],
              render_width=table_w,
              render_height=table_h,
              wxyz=(1.0, 0.0, 0.0, 0.0),
              position=table_positions[world],
              cast_shadow=False,
              receive_shadow=False,
            )
          )
          render_image_handles.append(
            scene.server.scene.add_image(
              f"/texture_demo/env_{world}/warp_render",
              blank,
              render_width=panel_w,
              render_height=panel_h,
              wxyz=panel_wxyz,
              position=offset + scene_offset + np.array([0.0, -1.00, 0.72], dtype=np.float32),
              cast_shadow=False,
              receive_shadow=False,
            )
          )
          label_handles.append(
            scene.server.scene.add_label(
              f"/texture_demo/env_{world}/label",
              f"env {world} | {texture_specs[int(selected_texture[world])].label}",
              position=offset + scene_offset + np.array([-0.55, -0.78, 0.52], dtype=np.float32),
              font_size_mode="scene",
              font_scene_height=0.055,
            )
          )

    def render_warp_images() -> None:
      mjw.refit_bvh(m, d, rc)
      mjw.render(m, d, rc)
      rgb_all = rc.rgb_data.numpy()
      rgb_adr = rc.rgb_adr.numpy()
      start = int(rgb_adr[camera_id])
      stop = start + args.width * args.height
      for world, handle in enumerate(render_image_handles):
        handle.image = _unpack_rgb(rgb_all[world, start:stop], args.width, args.height)

    def step_fn(_mjm, _mjd) -> None:
      nonlocal step_count
      if step_count % args.randomize_every == 0:
        set_table_textures()

      if mjm.nu:
        d.ctrl.assign(_control_targets(step_count, num_envs, ctrlrange))
      mjw.step(m, d)
      mjw.get_data_into(_mjd, _mjm, d, world_id=0)
      step_count += 1

    def render_fn(scene) -> None:
      nonlocal render_count
      ensure_viser_handles(scene)

      body_xpos = d.xpos.numpy() + env_offsets[:, None, :]
      mocap_pos = d.mocap_pos.numpy() + env_offsets[:, None, :]
      scene.update_from_arrays(
        body_xpos=body_xpos,
        body_xmat=d.xmat.numpy(),
        mocap_pos=mocap_pos,
        mocap_quat=d.mocap_quat.numpy(),
        qpos=d.qpos.numpy(),
        qvel=d.qvel.numpy(),
        ctrl=d.ctrl.numpy(),
      )
      update_custom_visual_positions(scene)

      if render_count % args.render_every == 0:
        render_warp_images()
      render_count += 1

    server = viser.ViserServer(port=args.port)
    print(
      f"Running texture randomization demo at http://localhost:{args.port}\n"
      f"  worlds: {num_envs}\n"
      f"  FR3 XML: {fr3_xml}\n"
      f"  textures compiled: {len(texture_specs)}\n"
      f"  texture reset interval: {args.randomize_every} physics steps\n"
      f"  Warp render resolution: {args.width}x{args.height}"
    )
    set_table_textures()
    MjViserViewer(mjm, mjd, step_fn=step_fn, render_fn=render_fn, num_envs=num_envs, server=server).run()


if __name__ == "__main__":
  main()
