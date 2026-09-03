# Workload definitions

`translate-segment.yaml` routes one bounded text segment to a pinned local
CUDA LibreTranslate service (capability `libretranslate`). Each worker host
runs its own loopback-only service on `127.0.0.1:5001` and advertises the
capability plus the `libretranslate-1.9.6-bounded-sbd` model pack; any number
of heterogeneous workers may serve the same route concurrently. Reviewed 8 GB
GTX 1070/1080-class Pascal cards are eligible at cc 6.1, but lack the RTX
Tensor Core fast path and must be benchmarked as a separate economics tier:

```sh
dgpu agent --endpoint https://gpu-control.example.com --agent-id host-a \
  --capability libretranslate --model-pack libretranslate-1.9.6-bounded-sbd
```

Jobs choose only a schema-bounded `text`, `source`, and `target` pair; the
endpoint and wire format are fixed by the route. For bulk use
`translate-text-batch.yaml` with capability `libretranslate-batch`: it accepts
an ordered same-language `segments` array and amortizes claims, HTTP setup,
artifacts, and state transitions. It remains model-server agnostic and contains
no producer-specific identifiers. `translate-batch.yaml` is the separate
true CTranslate2/Argos token-batching contract for audited model-pack workers.

`image-generate.yaml` is executable: it calls the bundled fixed SDXL
Diffusers plugin. A real image worker must additionally install the pinned
Diffusers runtime and download the exact model revision declared by the route.

The transcription and Qwen files are admission, scheduling, artifact, cost,
and scaling contract scaffolds. This repository does not bundle large model
weights or grant their licenses.

An agent must advertise a capability and model pack only after its image has:

- installed the named plugin handler;
- pinned and verified the model files;
- started and warmed any local model server;
- passed a canary for the declared input/output contract;
- measured initial throughput for its GPU class.

The SDXL executor keeps one pipeline cached per GPU and accepts no checkpoint,
LoRA, Python handler, or workflow graph from a job. ComfyUI is not required for
this route. It can be added later as a separate adapter when allowlisted graph
workflows, ControlNet, or multi-stage pipelines justify its larger operational
surface. The Qwen pack will translate the bounded batch contract into a warm
vLLM request and validate one response per prompt.
