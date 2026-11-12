# Rootless Docker sandbox

Agent code and verification commands run in a rootless Docker container by default.
The runner fails closed: missing Docker, a non-rootless daemon, or a missing local
image returns an execution error. It never silently executes the same command on
the host.

## 1. Install and enable rootless Docker

Install Docker Engine and follow Docker's official rootless mode setup for the
current Linux user. Start the rootless daemon and verify:

```bash
docker info --format '{{json .SecurityOptions}}'
```

The output must contain `rootless`. The application enforces this by default.
Do not point this configuration at a rootful Docker daemon.

## 2. Build the sandbox image

Run from the repository root:

```bash
docker build -f docker/sandbox.Dockerfile -t agentics-sandbox:py311 .
```

The build context excludes Git history, account cookies, `.env`, runtime data,
and local virtual environments through `.dockerignore`. The image contains the
project's Python dependencies, pytest, and Git for repository tests.

## 3. Configure and run

The defaults are enabled and conservative:

| Setting | Default | Purpose |
|---|---:|---|
| `SANDBOX_ENABLED` | `true` | Use Docker for all workspace commands |
| `SANDBOX_IMAGE` | `agentics-sandbox:py311` | Prebuilt local image; no implicit pull |
| `SANDBOX_REQUIRE_ROOTLESS` | `true` | Reject rootful Docker daemons |
| `SANDBOX_CPU_MILLIS` | `1000` | CPU quota in millicores |
| `SANDBOX_MEMORY_MB` | `1024` | Container memory and swap limit |
| `SANDBOX_PIDS_LIMIT` | `128` | Process limit |
| `SANDBOX_TMPFS_MB` | `128` | Temporary filesystem size |

The container has no network, a read-only root filesystem, dropped capabilities,
no-new-privileges, private IPC, a PID limit, and bounded CPU/memory. Only the
agent's own repository workspace is bind-mounted read-write. No GitHub token,
Perplexity cookie, LLM key, host home directory, or Docker socket is mounted or
passed into the container.

`SANDBOX_ENABLED=false` is an explicit opt-out for trusted local development
only. It restores the host-process runner and is not suitable for untrusted code.

## 4. Operational boundaries

- Rootless Docker reduces daemon/container privilege; it is not a complete VM boundary.
- The workspace is writable by design so tests can create files. Treat it as
  disposable and never place secrets in the repository workspace.
- Network access is disabled inside the sandbox. Research and GitHub API calls
  must remain in the trusted orchestrator process, outside the code container.
- The image must be rebuilt when project dependencies change. The runner does
  not pull arbitrary images or execute build steps on behalf of agents.
- Production deployments should add host-level disk quotas and monitoring for
  rootless daemon storage, and consider a dedicated VM per trust domain.
