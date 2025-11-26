# DLSS usage FAQ

## Does render resolution change visual detail or only the output size?
instant-ngp renders into a `full_resolution` surface whose size comes from the window or from the `--width/--height` arguments you pass to `scripts/run.py`, so those numbers set the DLSS output resolution and the size of the image written to disk.【F:scripts/run.py†L392-L398】【F:src/testbed.cu†L3256-L3307】

Every frame the renderer keeps an `in_resolution` that represents the DLSS input (the "internal shading" resolution). Without DLSS—or whenever dynamic resolution is disabled—the render buffer resizes both resolutions to the same value and the CUDA kernels trace exactly the pixels you requested.【F:src/render_buffer.cu†L600-L672】 When DLSS is active, the dynamic-resolution loop computes a scale factor, resizes the input buffer accordingly, and then asks DLSS to upsample that result into the fixed output surface. That is why you keep the requested output size even though DLSS can adjust how many rays are traced internally.【F:src/testbed.cu†L3333-L3382】

### What resolution is traced when I run `python scripts/run.py --video_camera_path <path> --width 3840 --height 2160 --dlss`?
1. The command-line dimensions populate `view.full_resolution = (3840, 2160)` for every render view.【F:scripts/run.py†L392-L409】【F:src/testbed.cu†L3256-L3307】
2. The dynamic-resolution loop starts from the previous frame’s input resolution (or `full_resolution` on the very first frame) and measures how many pixels were shaded vs. the output surface to build `pixel_ratio = traced_pixels / full_pixels`.【F:src/testbed.cu†L3333-L3361】
3. Using the last frame time it computes `factor = sqrt(pixel_ratio / frame_ms * 1000 / target_fps)`; this is the relative scale between the output size and the internal shading grid.【F:src/testbed.cu†L3333-L3359】
4. DLSS clamps the resulting resolution to the range it supports for the currently selected quality preset, so a Performance preset will allow a lower input size than Quality or DLAA. The final `in_resolution` therefore lands on the closest DLSS-compatible size at or below 3840×2160 depending on the preset and the measured frame time.【F:src/dlss.cu†L1074-L1156】【F:src/testbed.cu†L3361-L3374】

In other words, the 3840×2160 numbers you pass on the command line are always the DLSS **output** resolution. DLSS may render fewer internal pixels when it needs more performance, but the resulting frame is still upsampled into a 4K surface before it is written to disk.

### How does `fixed_res_factor` influence the input resolution?
Dynamic resolution relies on the scaling factor above; disabling it lets you pin a specific input size. Setting `testbed.dynamic_res = False` (or turning off the GUI toggle) replaces the adaptive scale with `factor = 8 / fixed_res_factor` for every frame.【F:src/testbed.cu†L3333-L3359】 The slider exposed in the GUI and Python API stores that value in `m_fixed_res_factor`:

* `fixed_res_factor = 8` ⇒ `factor = 1.0`, so `in_resolution == full_resolution` (no DLSS upscaling).
* `fixed_res_factor = 16` ⇒ `factor = 0.5`, so the input grid is half-resolution in each dimension and DLSS upscales by 2×.
* Larger factors follow the same pattern—the scaler divides 8 by your chosen number to decide how aggressively to reduce the traced resolution before DLSS upsamples it.

Requesting the `dlaa` preset automatically disables dynamic resolution and forces `fixed_res_factor = 8` so DLSS runs as a pure anti-aliasing pass at 1:1 input/output resolution.【F:src/testbed.cu†L5697-L5718】【F:scripts/run.py†L162-L175】

## Which DLSS quality preset is used, and how can I change it?
By default instant-ngp lets NGX choose the preset whose supported resolution range contains the current DLSS input size; as the scaler changes resolution, `Dlss::update_feature` selects whichever preset matches that input and reports the active mode in the GUI.【F:src/dlss.cu†L1074-L1156】【F:src/testbed.cu†L1403-L1435】 If you want a specific preset, use the new controls:

* **Command line:** pass `--dlss_mode <preset>` to `scripts/run.py`. Options include `ultra_performance`, `max_performance`, `balanced`, `max_quality`, `ultra_quality`, and `dlaa`. The script forwards the request through `testbed.dlss_mode`, which throws a clear error if the NGX runtime cannot provide that preset at the requested resolution.【F:scripts/run.py†L162-L175】【F:src/python_api.cu†L713-L728】
* **GUI:** when DLSS is active, the “DLSS mode” combo box lets you switch between `Auto` (the default heuristic) and any preset reported by NGX. Selecting a preset forces the same override as the command line, and `Auto` returns control to the dynamic-resolution chooser.【F:src/testbed.cu†L1403-L1435】

Use the `dlaa` preset to run DLSS as an anti-aliasing pass; the testbed disables dynamic resolution automatically so the input resolution equals the output before DLSS is evaluated.【F:src/testbed.cu†L5697-L5718】

## How do I make DLSS upscale to 4K?
DLSS always targets the render buffer's "full" resolution, which instant-ngp derives from the window or from the dimensions you pass on the command line.【F:src/testbed.cu†L3220-L3359】 To get a 3840×2160 DLSS output, request that size when you call the Python helper:

```bash
python scripts/run.py --video_camera_path <path> --width 3840 --height 2160 --dlss
```

`scripts/run.py` forwards those dimensions through its DLSS-aware capture helper, so DLSS will clamp the internal input resolution to a DLSS-supported size, run the upsampler, and finally splat the 4K output surface that gets written to disk.【F:scripts/run.py†L360-L472】【F:src/testbed.cu†L3333-L3382】 If you use the GUI, simply resize the window to 3840×2160 before toggling DLSS; the testbed records that window size as the DLSS output target.【F:src/testbed.cu†L3256-L3307】

## How does DLSS interact with offline rendering?
DLSS remains tied to the Vulkan-backed render window. That means offline pipelines still need a GLFW window alive before NGX initialization—matching the maintainer guidance to create a window, set a smaller render resolution, disable dynamic resolution, enable DLSS, and drive frames via `testbed.frame()` plus `testbed.screenshot()` rather than `testbed.render()`. The Python helpers already honor this requirement: when `--dlss` is passed they create a hidden window, initialize Vulkan/NGX, lock `full_resolution` to the requested output size, and—unlike windowless renders—capture each frame by alternating `frame()` and `screenshot()` so headless exports stay on the DLSS-enabled windowed path.【F:scripts/run.py†L86-L209】【F:scripts/run.py†L360-L472】【F:src/testbed.cu†L3256-L3382】

If you want to mimic the maintainer’s 1080p→4K example, launch a 3840×2160 window (or pass `--width 3840 --height 2160`), disable dynamic resolution, set `fixed_res_factor = 16` so the input grid is 0.5× each dimension, and enable DLSS. The DLSS pass will then upscale that 1080p input into the 4K output surface for each `frame()`/`screenshot()` pair.【F:src/testbed.cu†L3333-L3374】【F:src/render_buffer.cu†L600-L672】 If you keep dynamic resolution enabled instead, the scaler may lower the input size further when frame time increases, so pinning the factor ensures a predictable internal resolution while offline rendering.
