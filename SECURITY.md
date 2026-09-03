# Security Policy

## Supported versions

Only the latest commit on the default branch receives security updates. The
project is currently at an early release stage; no long-term support branches
exist yet.

## Reporting a vulnerability

Please report security vulnerabilities privately rather than opening a public
issue. This gives maintainers time to release a fix before details are public.

Options:

- Open a draft security advisory on the repository if GitHub private
  vulnerability reporting is enabled.
- Contact the maintainers directly through a private channel.

Do not include production credentials, tokens, logs, or hostnames in any
security report. If you need to share an artifact, redact or replace private
identifiers with neutral example values.

## What to report

We are especially interested in:

- Authentication or authorization bypasses.
- Leaks of agent, client, or provider API credentials.
- Ways for a job payload to escape its trusted route (for example, selecting
  arbitrary code, container images, model files, or workflow graphs).
- Bypasses of artifact validation, digest checks, or output bounds.
- Provider lifecycle or autoscaler spend-safety issues.

## Security design notes

- Workload routes are the trust boundary. A job may supply only schema-bounded
  data; it must never select arbitrary code, model files, container images,
  LoRAs, or workflow graphs.
- Provider API keys never leave the private controller process.
- Worker tokens are short-lived and scoped to specific routes.
- Outputs are content-addressed and validated against the declared contract.

## Acknowledgments

We will credit reporters who disclose vulnerabilities responsibly, unless they
prefer to remain anonymous.
