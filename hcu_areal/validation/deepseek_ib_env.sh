#!/usr/bin/env bash
# Source AFTER common_env.sh, on each of the two selected nodes, before Ray.
export LD_LIBRARY_PATH="/opt/rccl-rdma-sharp-plugins/lib:${LD_LIBRARY_PATH:-}"
export NCCL_NET=IB NCCL_PLUGIN_P2P=IB NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
_deepseek_interfaces=$("${PYTHON_BIN:-python3}" -S - <<'PY'
import fcntl
import ipaddress
from pathlib import Path
import socket
import struct

interfaces=[]
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as connection:
    for _,name in socket.if_nameindex():
        try:
            raw=fcntl.ioctl(connection.fileno(),0x8915,struct.pack('256s',name.encode()[:15]))
            if socket.inet_ntoa(raw[20:24]).startswith('10.16.1.'):
                interfaces.append(name)
        except OSError:
            pass
ports=[]
for device in Path('/sys/class/infiniband').glob('*'):
    for port in (device/'ports').glob('*'):
        try:
            address=ipaddress.IPv6Address((port/'gids/3').read_text().strip()).ipv4_mapped
            if ((port/'state').read_text().startswith('4:')
                    and (port/'gid_attrs/types/3').read_text().strip()=='RoCE v2'
                    and address and str(address).startswith('12.12.12.')):
                ports.append(device.name+':'+port.name)
        except (OSError,ValueError):
            pass
if len(interfaces)!=1 or len(ports)!=1:
    raise SystemExit('Expected one 10.16.1.* interface and one active 12.12.12.* RoCE v2 GID3 port: '+repr((interfaces,ports)))
print(interfaces[0])
print(ports[0])
PY
) || return 2
_deepseek_interface=${_deepseek_interfaces%%$'\n'*}
_deepseek_hca=${_deepseek_interfaces##*$'\n'}
export NCCL_SOCKET_IFNAME="=$_deepseek_interface" GLOO_SOCKET_IFNAME="$_deepseek_interface"
export NCCL_IB_HCA="=$_deepseek_hca"
if [[ -n "${RCCL_LOG_DIR:-}" ]]; then
    mkdir -p "$RCCL_LOG_DIR"
    export NCCL_DEBUG_FILE="$RCCL_LOG_DIR/log.%h.%p"
fi
unset _deepseek_interfaces _deepseek_interface _deepseek_hca
