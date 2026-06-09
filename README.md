# mujoco_textures

`mujoco_textures` is a small Python package plus a texture asset checkout for MuJoCo and MuJoCo Warp domain randomization demos. It includes converted texture images, a manifest with source metadata, a downloader that can regenerate the texture set, and a Viser demo for per-world material texture randomization.

The package is intended to be useful from downstream environments that compile many candidate textures once and then randomize material texture IDs at reset time.

## Layout

```text
mujoco_textures/
  textures.py                    # manifest loading and sampling helpers
  download_textures.py           # downloader/converter CLI
  randomize_textures_viser.py    # MuJoCo Warp + Viser demo
textures/
  manifest.json
  robosuites/
  lerobot-libero/
  isaac-lib/
  polyhaven/
scripts/
  download_textures.py
  randomize_textures_viser.py
```

The checked-in texture files are converted to square RGB PNGs. The current manifest contains 954 textures:

| Source id | Folder | Count | Upstream |
| --- | --- | ---: | --- |
| `robosuite` | `textures/robosuites` | 30 | `ARISE-Initiative/robosuite`, `robosuite/models/assets/textures` |
| `lerobot_libero_assets` | `textures/lerobot-libero` | 121 | Hugging Face dataset `lerobot/libero-assets` |
| `nvidia_isaac_materials` | `textures/isaac-lib` | 45 | Texture references discovered from `NVlabs/GR00T-VisualSim2Real` IsaacSim config and material files |
| `polyhaven` | `textures/polyhaven` | 758 | Poly Haven texture diffuse/albedo maps |

Every manifest entry records the converted file path, source path, source URL, license label, source hash, and converted file hash.

## Install

From this checkout:

```bash
uv sync
```

For the visualization demo:

```bash
uv sync --extra visualize
```

The Viser demo needs MuJoCo Warp with per-world `mat_texid` support.

Until that support is available in the MuJoCo Warp release you are using, install the feature branch in the same environment:

```bash
uv pip install -e ~/code/mujoco_warp
```

## List And Sample Textures

```bash
uv run mujoco-textures-list
uv run mujoco-textures-list --source robosuite --paths
```

From Python:

```python
from mujoco_textures import load_manifest, sample_textures

manifest = load_manifest()
textures = sample_textures(10, seed=0)

for texture in textures:
    print(texture.name, texture.file, texture.source)
```

## Use With MuJoCo Specs

The helper returns resolved image paths that can be added to a `mujoco.MjSpec` before compilation:

```python
import mujoco
from mujoco_textures import sample_textures

spec = mujoco.MjSpec()
textures = sample_textures(10, seed=7)

for texture in textures:
    spec.add_texture(
        name=texture.name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=texture.file.as_posix(),
    )

spec.add_material(
    name="table_mat",
    textures=["", textures[0].name],
    texrepeat=[7, 5],
    texuniform=False,
)
```

With MuJoCo Warp per-world texture IDs, downstream code can sample among those compiled textures without recompiling the model:

```python
import numpy as np
import mujoco
import mujoco_warp as mjw

nworld = 4
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=nworld)

mjw.expand_mat_texid(m, nworld)

table_mat = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
texture_ids = np.array([
    mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_TEXTURE, texture.name)
    for texture in textures
])

mat_texid = m.mat_texid.numpy()
mat_texid[:, table_mat, rgb_role] = np.random.choice(texture_ids, size=nworld)
m.mat_texid.assign(mat_texid)
```

The renderer resolves `mat_texid[world_id, material_id, texture_role]`, so each world can render the same material with a different texture.

## Visualize Randomization

Run the default four-world demo:

```bash
uv run mujoco-textures-viser --num-envs 4 --randomize-every 50
```

The same entry point is also available as a script path:

```bash
uv run scripts/randomize_textures_viser --num-envs 4 --randomize-every 50
```

The demo uses `benchmarks/franka_emika_panda/panda.xml` automatically if it finds a local `mujoco_warp` checkout at `~/code/mujoco_warp`. You can also pass it explicitly:

```bash
uv run mujoco-textures-viser \
  --panda-xml ~/code/mujoco_warp/benchmarks/franka_emika_panda/panda.xml \
  --source robosuite \
  --max-textures 32
```

If no Panda XML is available, the demo still runs a standalone table scene so the texture randomization path can be inspected.

## Regenerate Textures

```bash
uv run mujoco-textures-download --output-root textures
```

Useful options:

```bash
uv run mujoco-textures-download --output-root textures --skip-polyhaven
uv run mujoco-textures-download --output-root textures --polyhaven-limit 100
uv run mujoco-textures-download --output-root textures --size 512 --workers 32
```

The downloader discovers images from the upstream sources, center-crops them, resamples them to square RGB PNGs, places them into the source-specific folders, and rewrites `textures/manifest.json`.

## Texture Source Notes

The code in this repository is Apache-2.0. Texture files retain their upstream license or usage terms; check `textures/manifest.json` before redistributing a subset.

The source labels currently used by the manifest are:

- robosuite textures: MIT.
- LeRobot LIBERO assets: the Hugging Face dataset metadata does not currently declare a concrete license; the manifest preserves the dataset source URL and labels those entries as dataset-license assets.
- NVIDIA Isaac material textures: discovered from the GR00T-VisualSim2Real IsaacSim configuration and referenced material files; manifest entries use the NVIDIA Omniverse asset license label from the downloader.
- Poly Haven textures: CC0-1.0.
