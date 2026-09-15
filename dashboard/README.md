# Local live dashboard

Start the read-only dashboard from the repository root:

```sh
python scripts/serve_dashboard.py --port 8765
```

Open <http://127.0.0.1:8765/>. The page polls the safe snapshot every three
seconds. Use `--status PATH` to serve a different snapshot. The server exposes
only the dashboard, `/status.json`, the comparison page at `/compare`, and a
fixed allowlist of before/after experiment artifacts. It does not expose the
repository tree.

The comparison page displays raw prompts and responses from the saved base and
final-candidate runs. Harmful prompt text is collapsed by default. All content
stays on the local machine.

Refresh the snapshot once, or keep it updated while a remote run is active:

```sh
python scripts/update_dashboard_status.py --config .dashboard-runtime.json
python scripts/update_dashboard_status.py --config .dashboard-runtime.json --watch --interval 5
```

`.dashboard-runtime.json` is ignored by Git. It may contain a host, SSH port,
identity-file path, remote JSONL output paths, and a process pattern. Do not put
credentials, prompts, generations, or raw model output in it. The updater uses
`wc -l` over SSH, `nvidia-smi`, a process-presence check, and the Vast CLI's
numeric `credit` field only. Writes use a same-directory temporary file and
`os.replace`, so the browser never observes a partial JSON document.

The optional ignored `overlay_file` adds safe aggregate results at each poll.
This lets scoring update the page without changing the remote polling process.
