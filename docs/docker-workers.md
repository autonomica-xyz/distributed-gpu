# Docker GPU workers

This is the copy/paste **owned-host integration** path for one trusted workload
service and one DGPU agent on a Docker GPU host. The example assigns GPU 0 to
the bounded LibreTranslate route. Run one model-service/agent pair per
physical GPU; never start multiple agents that advertise the same GPU.

For Salad/Vast production, publish both reviewed images to a trusted registry
and reference immutable digests. LibreTranslate's first-start package download
must be replaced by a build-time audited, hash-verified, pre-baked model pack;
do not let an untrusted provider worker populate models from the package index
at runtime.

## 1. Build the reviewed images

From the controller repository, copy the small build definitions and build on
the worker:

```sh
WORKER=operator@worker
ssh "$WORKER" 'rm -rf /tmp/dgpu-build && mkdir -p /tmp/dgpu-build'
scp -r deploy/agent deploy/libretranslate "$WORKER":/tmp/dgpu-build/
ssh "$WORKER" '
  docker build -t dgpu-agent:0.1.0 /tmp/dgpu-build/agent
  docker build -t dgpu-libretranslate:1.9.6-bounded-sbd \
    /tmp/dgpu-build/libretranslate
'
```

Both Dockerfiles pin their base image by digest. The pinned LibreTranslate
image uses Python 3.10; the DGPU wheel supports >=3.10 and is smoke-tested in
that image to prevent provider restart loops from metadata incompatibility.
The agent build downloads the public wheel and rejects it unless its SHA-256 matches the reviewed value in
`deploy/agent/Dockerfile`. Production images should be published to a trusted
registry and then referenced by their resulting immutable digest.

## 2. Issue and transfer a scoped token

Run on the controller. The raw credential travels over SSH stdin and is never
placed in a Docker argument or printed:

```sh
cd /opt/dgpu
WORKER=operator@worker
AGENT_ID=worker-translate-01
TOKEN_ID=agent-$AGENT_ID
TOKEN=$(.venv/bin/dgpu token \
  --role agent --subject "$AGENT_ID" \
  --scope routes=libretranslate \
  --expires-hours 168 --token-id "$TOKEN_ID")

printf 'DGPU_TOKEN=%s\n' "$TOKEN" | ssh "$WORKER" \
  "umask 077; mkdir -p ~/.config/dgpu; cat > ~/.config/dgpu/$AGENT_ID.env"
unset TOKEN
```

A worker gets only its identity and a short-lived route-scoped token. Provider
keys and operator/client tokens never belong on the worker.

## 3. Start the loopback model service

Run on the worker:

```sh
AGENT_ID=worker-translate-01
GPU=0
MODEL=dgpu-$AGENT_ID-model
MODEL_VOLUME=dgpu-$AGENT_ID-models

docker volume create "$MODEL_VOLUME"
docker run -d --name "$MODEL" \
  --gpus "device=$GPU" --memory 12g --restart unless-stopped \
  -v "$MODEL_VOLUME":/home/worker \
  -p 127.0.0.1:5001:5000 \
  dgpu-libretranslate:1.9.6-bounded-sbd \
  --load-only en,ar,bg,bn,ca,cs,da,de,el,es,fa,fr,he,hi,hu,id,it,ja,ko,lv,nl,pl,pt,ro,ru,sl,sq,sv,sw,th,tr,uk,vi,zh \
  --threads 8 --disable-web-ui --disable-files-translation \
  --char-limit 20000 --batch-limit 64 --req-limit 1200

# On an owned integration host, first start may populate the model volume.
# Wait until this works:
curl --fail http://127.0.0.1:5001/languages
```

Port 5001 is loopback-only. Never publish LibreTranslate on `0.0.0.0`.

## 4. Start the DGPU agent

The agent uses host networking only to reach the loopback model service; it
does not listen on a port.

```sh
AGENT_ID=worker-translate-01
GPU=0
AGENT=dgpu-$AGENT_ID-agent

docker volume create "$AGENT-work"
docker run -d --name "$AGENT" \
  --gpus "device=$GPU" --network host --restart unless-stopped \
  --env-file "$HOME/.config/dgpu/$AGENT_ID.env" \
  -v "$AGENT-work":/work \
  dgpu-agent:0.1.0 \
  --endpoint https://gpu-control.example.com \
  --agent-id "$AGENT_ID" \
  --provider local --provider-resource-id "$(hostname)" \
  --capability libretranslate \
  --model-pack libretranslate-1.9.6-bounded-sbd \
  --label pool=translate-live --label "host=$(hostname)" \
  --workdir /work --heartbeat-seconds 15 \
  --lease-seconds 120 --poll-seconds 3

docker logs -f "$AGENT"
```

The GPU should appear in the private dashboard within one heartbeat interval.
The command works unchanged on non-tailnet hosts because the agent calls the
public authenticated API at `gpu-control.example.com`.

## Stop and revoke

Stop the worker before retiring its credential:

```sh
# Worker host
AGENT_ID=worker-translate-01
docker rm -f "dgpu-$AGENT_ID-agent" "dgpu-$AGENT_ID-model"
rm -f "$HOME/.config/dgpu/$AGENT_ID.env"
# Keep the model volume for fast restart, or explicitly remove it:
# docker volume rm "dgpu-$AGENT_ID-models" "dgpu-$AGENT_ID-agent-work"

# Controller
cd /opt/dgpu
.venv/bin/dgpu revoke-token --token-id "agent-$AGENT_ID"
```

Use a new identity/token ID for every physical worker. Do not copy one env file
across hosts.
