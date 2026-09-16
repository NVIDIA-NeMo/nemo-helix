#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Outcome verifier: the counts file matches exactly and the inputs are untouched.
mkdir -p /logs/verifier

expected=$'a.txt: 4\nb.txt: 5\nc.txt: 1'
actual="$(sed -e 's/[[:space:]]*$//' /app/word_counts.txt 2>/dev/null | grep -v '^$')"
inputs_intact=1
[ "$(cat /app/notes/a.txt)" = $'alpha beta gamma\ndelta' ] || inputs_intact=0
[ "$(cat /app/notes/b.txt)" = 'one two three four five' ] || inputs_intact=0
[ "$(cat /app/notes/c.txt)" = 'solo' ] || inputs_intact=0

if [ "$actual" = "$expected" ] && [ "$inputs_intact" = 1 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
