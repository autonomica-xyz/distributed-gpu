# Private deployment configuration

This directory is for configuration that should never be committed.

- `private.env` — real runtime values for your own infrastructure (domains,
  listener addresses, provider organization, historical worker/group names, etc.).
  It is ignored by Git. Copy it to `/etc/dgpu/controller.env` for the systemd
  private service, or source it in a shell:

  ```bash
  set -a
  source config/private.env
  set +a
  ```

- `secrets/` — put actual API keys, tokens, and key files here. This directory is
  also ignored by Git.

The committed examples in `deploy/systemd/dgpu-controller.env.example`,
`deploy/agent/Dockerfile`, and the `*.example.yaml` pool files use neutral
placeholders so the repository can be published. When you build or deploy for
your own setup, fill in `config/private.env` and apply the values to the
relevant files.
