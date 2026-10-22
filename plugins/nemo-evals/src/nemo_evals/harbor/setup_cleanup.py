# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell scripts that close an agent setup egress window: list processes before it opens, then kill new
processes still holding an outside connection once the setup hosts are blocked again.

Egress sidecars and providers keep established connections across a policy change, so blocking the hosts
alone would let a process setup started keep talking to them. Both scripts need ``/bin/sh`` and ``/proc``
in the task image and run as root with :data:`SYSTEM_PATH`.
"""

SHELL = "/bin/sh"
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# A process's identity is "<pid>:<start time>", so a PID reused later isn't mistaken for the process
# that had it. Fields are read after the last ")", because a command name may contain spaces.
_PROCESS_IDENTITY_FUNCTIONS = """\
ident() {
  id=
  state=
  read -r line < "/proc/$1/stat" 2>/dev/null || return 1
  ident_pid=$1
  set -- ${line##*) }
  state=$1
  parent=$2
  id="$ident_pid:${20}"
}
"""

# Prints one identity per running process, except this command's own shell.
LIST_PROCESSES_SCRIPT = (
    _PROCESS_IDENTITY_FUNCTIONS
    + """\
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ "$pid" = "$$" ] && continue
  ident "$pid" && echo "$id"
done
"""
)

# Kills every process started since $KEEP was listed that still holds a connection to a non-loopback
# address, except this shell and its ancestors (the execd chain). Once the setup hosts' rules are gone
# nothing can open a new connection to them, so only these can still reach one; processes an agent's
# setup leaves for its run (tmux, a desktop session) hold none and survive. A process whose descriptors
# can't be read, or every new process when `readlink` is missing, is killed as a holder; the sweep's own
# children (`readlink`) are skipped like its ancestors. Succeeds only once a sweep finds no holder alive,
# so forked children that inherit a socket are caught too.
KILL_CONNECTION_HOLDERS_SCRIPT = (
    _PROCESS_IDENTITY_FUNCTIONS
    + """\
outside_sockets() {
  sockets=" "
  for table in /proc/net/tcp /proc/net/tcp6 /proc/net/udp /proc/net/udp6; do
    [ -r "$table" ] || continue
    while read -r _ _ remote state _ _ _ _ _ inode _; do
      case "$remote" in *:*) ;; *) continue ;; esac
      [ "${remote##*:}" = 0000 ] && continue
      [ "$state" = 0A ] && continue
      [ "$inode" = 0 ] && continue
      case "${remote%:*}" in
        ??????7F | 0000000000000000FFFF0000??????7F | 00000000000000000000000001000000) continue ;;
      esac
      sockets="$sockets$inode "
    done < "$table"
  done
}
holds_outside_socket() {
  [ "$inspect" = 1 ] || return 0
  [ -r "/proc/$1/fd" ] || return 0
  for fd in /proc/$1/fd/*; do
    if ! target=$(readlink "$fd" 2>/dev/null); then
      [ -L "$fd" ] && return 0
      continue
    fi
    case "$target" in
      "socket:["*"]")
        inode=${target#socket:[}
        case "$sockets" in *" ${inode%]} "*) return 0 ;; esac
        ;;
    esac
  done
  return 1
}
if command -v readlink > /dev/null 2>&1; then inspect=1; else inspect=0; fi
ancestors=" "
p=$$
while [ -n "$p" ] && [ "$p" != 0 ]; do
  ancestors="$ancestors$p "
  read -r line < "/proc/$p/stat" 2>/dev/null || break
  set -- ${line##*) }
  p=$2
done
passes=0
while :; do
  outside_sockets
  alive=
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    case "$ancestors" in *" $pid "*) continue ;; esac
    ident "$pid" || continue
    case " $KEEP " in *" $id "*) continue ;; esac
    [ "$state" = Z ] && continue
    [ "$parent" = "$$" ] && continue
    holds_outside_socket "$pid" || continue
    alive="$alive $pid"
    kill -9 "$pid" 2>/dev/null
  done
  [ -z "$alive" ] && exit 0
  passes=$((passes + 1))
  if [ "$passes" -ge 50 ]; then
    echo "could not kill:$alive" >&2
    exit 1
  fi
  sleep 0.1 2>/dev/null || sleep 1
done
"""
)
