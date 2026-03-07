#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.24.3}"
DEST_DIR="${1:-ext/onnx_ruby/onnxruntime}"

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Darwin)
      case "$arch" in
        arm64) echo "osx-arm64" ;;
        x86_64) echo "osx-x86_64" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    Linux)
      case "$arch" in
        x86_64|amd64) echo "linux-x64" ;;
        aarch64|arm64) echo "linux-aarch64" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    *) echo "unsupported OS: $os" >&2; exit 1 ;;
  esac
}

PLATFORM="$(detect_platform)"
FILENAME="onnxruntime-${PLATFORM}-${ORT_VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${FILENAME}"

echo "Downloading ONNX Runtime v${ORT_VERSION} for ${PLATFORM}..."

mkdir -p "$DEST_DIR"

if command -v curl &>/dev/null; then
  curl -fSL "$URL" -o "/tmp/${FILENAME}"
elif command -v wget &>/dev/null; then
  wget -q "$URL" -O "/tmp/${FILENAME}"
else
  echo "Error: curl or wget required" >&2
  exit 1
fi

echo "Extracting to ${DEST_DIR}..."
tar xzf "/tmp/${FILENAME}" -C "$DEST_DIR" --strip-components=1
rm -f "/tmp/${FILENAME}"

echo "ONNX Runtime v${ORT_VERSION} installed to ${DEST_DIR}"
