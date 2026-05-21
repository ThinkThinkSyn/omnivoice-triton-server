#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVICE_NAME="omnivoice-server"
CUDA_VISIBLE_DEVICES_VALUE=""
PYTHON_BIN="${OMNIVOICE_PYTHON:-$(command -v python)}"
SERVICE_USER="${OMNIVOICE_SERVICE_USER:-$(id -un)}"
SERVICE_GROUP="${OMNIVOICE_SERVICE_GROUP:-$(id -gn)}"
WORKING_DIR="$ROOT"
ENABLE_SERVICE=1
START_SERVICE=1
EXTRA_ENV=()
APP_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/install_systemd_service.sh --cuda-visible-devices IDS [options] -- [omnivoice launcher args...]

Required:
  --cuda-visible-devices IDS       Value for CUDA_VISIBLE_DEVICES, e.g. 0 or 6,7.
  --cuda_visible_devices IDS       Alias for --cuda-visible-devices.

Options:
  --service-name NAME              systemd unit name without .service. Default: omnivoice-server
  --python PATH                    Python executable. Default: current python or OMNIVOICE_PYTHON.
  --user USER                      Linux user for the service. Default: current user.
  --group GROUP                    Linux group for the service. Default: current group.
  --working-dir PATH               Repository root. Default: this repository.
  --env KEY=VALUE                  Extra environment line. Can be repeated.
  --no-enable                      Do not enable service at boot.
  --no-start                       Do not restart service after installation.
  -h, --help                       Show this help.

Example:
  scripts/install_systemd_service.sh \
    --cuda-visible-devices 6,7 \
    --python /home/server10/miniconda3/envs/omnivoice/bin/python \
    -- \
    --port 9194 \
    --model-id /home/server10/omnivoice/models/OmniVoice \
    --gpu-inferer 2 \
    --max-batch-size 16
EOF
}

shell_quote() {
  printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-visible-devices|--cuda_visible_devices)
      CUDA_VISIBLE_DEVICES_VALUE="${2:?missing value for --cuda-visible-devices}"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="${2:?missing value for --service-name}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --user)
      SERVICE_USER="${2:?missing value for --user}"
      shift 2
      ;;
    --group)
      SERVICE_GROUP="${2:?missing value for --group}"
      shift 2
      ;;
    --working-dir)
      WORKING_DIR="${2:?missing value for --working-dir}"
      shift 2
      ;;
    --env)
      EXTRA_ENV+=("${2:?missing value for --env}")
      shift 2
      ;;
    --no-enable)
      ENABLE_SERVICE=0
      shift
      ;;
    --no-start)
      START_SERVICE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      APP_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option before --: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
  echo "--cuda-visible-devices is required" >&2
  usage >&2
  exit 2
fi

if [[ ! "$SERVICE_NAME" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
  echo "Invalid --service-name: $SERVICE_NAME" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -d "$WORKING_DIR/src" ]]; then
  echo "Working directory does not look like this repo: $WORKING_DIR" >&2
  exit 2
fi

sudo -v

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

wrapper_path="/etc/omnivoice/${SERVICE_NAME}.sh"
service_path="/etc/systemd/system/${SERVICE_NAME}.service"
wrapper_tmp="$tmpdir/${SERVICE_NAME}.sh"
service_tmp="$tmpdir/${SERVICE_NAME}.service"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd $(shell_quote "$WORKING_DIR")"
  echo "export CUDA_VISIBLE_DEVICES=$(shell_quote "$CUDA_VISIBLE_DEVICES_VALUE")"
  echo "export PYTHONDONTWRITEBYTECODE=\${PYTHONDONTWRITEBYTECODE:-1}"
  echo "export PYTHONUNBUFFERED=\${PYTHONUNBUFFERED:-1}"
  echo "export PYTHONPATH=$(shell_quote "$WORKING_DIR/src")\${PYTHONPATH:+:\$PYTHONPATH}"
  for env_line in "${EXTRA_ENV[@]}"; do
    key="${env_line%%=*}"
    value="${env_line#*=}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid environment key in --env: $key" >&2
      exit 2
    fi
    echo "export $key=$(shell_quote "$value")"
  done
  echo 'APP_ARGS=('
  for arg in "${APP_ARGS[@]}"; do
    echo "  $(shell_quote "$arg")"
  done
  echo ')'
  echo "exec $(shell_quote "$PYTHON_BIN") -m omnivoice-triton-server start \"\${APP_ARGS[@]}\""
} > "$wrapper_tmp"

cat > "$service_tmp" <<EOF
[Unit]
Description=OmniVoice Triton Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$WORKING_DIR
ExecStart=$wrapper_path
Restart=always
RestartSec=5
TimeoutStopSec=90
KillSignal=SIGTERM
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

sudo install -d -m 0755 /etc/omnivoice
sudo install -m 0755 "$wrapper_tmp" "$wrapper_path"
sudo install -m 0644 "$service_tmp" "$service_path"
sudo systemctl daemon-reload

if [[ "$ENABLE_SERVICE" -eq 1 ]]; then
  sudo systemctl enable "${SERVICE_NAME}.service"
fi

if [[ "$START_SERVICE" -eq 1 ]]; then
  sudo systemctl restart "${SERVICE_NAME}.service"
fi

echo "Installed $service_path"
echo "Wrapper $wrapper_path"
echo "Status:"
sudo systemctl --no-pager --lines=20 status "${SERVICE_NAME}.service" || true
