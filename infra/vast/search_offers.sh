#!/usr/bin/env bash
set -euo pipefail

# Read-only. This command does not create an instance.
uvx --from vastai vastai --raw search offers \
  'rentable=true verified=true num_gpus=1 gpu_ram>=75 disk_space>=300 reliability>=0.98 inet_down>=500 cuda_vers>=12.8' \
  --storage 300 \
  --limit 30 \
  -o dph |
  jq '[.[] | {
    offer_id: .id,
    gpu: .gpu_name,
    gpu_ram_mb: .gpu_ram,
    total_hourly_usd: .dph_total,
    reliability: .reliability,
    disk_available_gb: .disk_space,
    disk_bandwidth_mb_s: .disk_bw,
    download_mbit_s: .inet_down,
    location: .geolocation
  }]'
