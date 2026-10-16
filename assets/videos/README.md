# Sparehour historical videos

This footage, its posters, and burned-in captions predate the Sparehour rebrand.
They preserve the previous product names and interface as recorded; see the
repository README screenshot for the current dashboard.

Real, captioned product demonstrations, edited for length. The footage uses
one self-hosted NVIDIA worker and actual dispatched SDXL jobs—not simulated
interfaces or metrics. Model weights and ML runtimes are provisioned
separately; these videos are not cold-start or throughput benchmarks.

| Video | Duration | Poster | Editable captions |
| --- | --- | --- | --- |
| [Controller: queue to delivered image](controller-demo.mp4) | 60 seconds | [PNG](controller-demo-poster.png) | [English SRT](controller-demo.srt) |
| [Sparehour + sparehour-worker overview](overview.mp4) | 45 seconds | [PNG](overview-poster.png) | [English SRT](overview.srt) |

The controller demonstration covers the live fleet and queue, schema-bound
job submission, leased GPU execution, artifact download, and the generated
image. The overview connects the controller with the worker's selection and
pause/resume controls. The [worker demonstration](https://github.com/autonomica-xyz/sparehour-worker/tree/main/assets/videos)
is published in the companion repository.

## Share or embed

- MP4: 1920×1080, 24 fps, H.264, browser-friendly fast start, no audio track.
- Captions are already burned into each MP4. SRT sidecars support
  accessibility, translation, and re-editing.
- Posters are 1920×1080 PNG images. Use a clickable poster linking to the
  MP4 in GitHub Markdown; use the poster with an HTML5 video player on a website.
- Videos contain no music or voiceover and work muted in a social feed.
- Keep the “edited for length” and separately provisioned model/runtime
  context when reusing footage; do not present the edits as benchmark timings.

## Licenses

Sparehour is GPL-3.0-or-later. The companion worker is MIT-licensed
and depends on the GPL-3.0-or-later controller package. Model weights have
separate license terms and are not included with these media assets.
