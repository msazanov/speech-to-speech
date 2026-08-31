#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="${RHVOICE_INSTALL_ROOT:-${repo_root}/models/rhvoice}"
package_dir="${install_root}/packages"
runtime_root="${install_root}/root"

for command_name in pacman pacman-key curl bsdtar; do
  command -v "${command_name}" >/dev/null || {
    echo "Missing required command: ${command_name}" >&2
    exit 1
  }
done

mkdir -p "${package_dir}" "${install_root}"
staging_dir="$(mktemp -d "${install_root}/.rhvoice-stage.XXXXXX")"
trap 'rm -rf -- "${staging_dir}"' EXIT
staging_root="${staging_dir}/runtime"
mkdir -p "${staging_root}"

packages=(
  rhvoice
  rhvoice-language-russian
  rhvoice-voice-aleksandr
  rhvoice-voice-mikhail
  rhvoice-voice-evgeniy-rus
  rhvoice-voice-pavel
)

for package_name in "${packages[@]}"; do
  package_url="$(pacman -Sp --print-format '%l' "${package_name}" | tail -n 1)"
  archive_path="${package_dir}/${package_name}.pkg.tar.zst"
  signature_path="${archive_path}.sig"
  curl -fL --retry 2 --output "${archive_path}" "${package_url}"
  curl -fL --retry 2 --output "${signature_path}" "${package_url}.sig"
  pacman-key --verify "${signature_path}" "${archive_path}"
  bsdtar -xf "${archive_path}" -C "${staging_root}"
done

test -x "${staging_root}/usr/bin/RHVoice-test"
test -d "${staging_root}/usr/share/RHVoice/languages/Russian"

runtime_name="runtime-$(date +%s)-${RANDOM}"
runtime_path="${install_root}/${runtime_name}"
mv -f "${staging_root}" "${runtime_path}"
ln -s "${runtime_name}" "${staging_dir}/root"
if [[ -e "${runtime_root}" && ! -L "${runtime_root}" ]]; then
  mv -f "${runtime_root}" "${install_root}/root.previous.$(date +%s)"
fi
mv -Tf "${staging_dir}/root" "${runtime_root}"

executable="${runtime_root}/usr/bin/RHVoice-test"
data_path="${runtime_root}/usr/share/RHVoice"
library_path="${runtime_root}/usr/lib"

printf '%s\n' \
  "RHVoice installed without root access." \
  "Executable: ${executable}" \
  "Data: ${data_path}" \
  "Libraries: ${library_path}"
