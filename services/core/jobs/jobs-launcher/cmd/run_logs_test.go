// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestRunExecDoesNotHangOnProgressBar(t *testing.T) {
	// 256KB of carriage returns and no newline. bufio.Scanner stops at 64KB,
	// tailOutput returns, and the child blocks forever on the unread pipe.
	const script = `dd if=/dev/zero bs=1024 count=256 2>/dev/null | tr '\0' '\r'; printf '\nfinished\n'`

	done := make(chan struct{})
	var exitCode int
	var runErr error
	go func() {
		defer close(done)
		exitCode, runErr = runExecWithStdin([]string{"sh", "-c", script})
	}()

	select {
	case <-done:
		if runErr != nil || exitCode != 0 {
			t.Fatalf("runExec exit=%d err=%v", exitCode, runErr)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("runExec hung tailing a carriage-return progress line longer than 64KB")
	}
}

func TestForEachLogLineCarriageReturnReplacesLine(t *testing.T) {
	// tqdm writes \r, a snapshot, padding spaces, then the next snapshot, and a
	// newline when the bar finishes. A carriage return overwrites the previous
	// snapshot, so only the one left in place at the newline is logged.
	const input = "\rDownloading:  50%|█████     | 1.87G  \rDownloading: 100%|██████████| 3.72G\nnext line\n"

	got := readLogLines(input)
	want := []string{
		"Downloading: 100%|██████████| 3.72G",
		"next line",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("lines = %#v, want %#v", got, want)
	}
}

func TestForEachLogLineCarriageReturnAtEOFEmitsSnapshot(t *testing.T) {
	// A progress line that ends on \r, with no newline before the process
	// exits, is still a line. The snapshot before that \r has to be logged.
	got := readLogLines("done\r")
	want := []string{"done"}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("lines = %#v, want %#v", got, want)
	}

	// An earlier snapshot is still discarded when a later one follows. Only
	// the snapshot left in place at EOF is logged.
	got = readLogLines("progress\rdone\r")
	want = []string{"done"}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("lines = %#v, want %#v", got, want)
	}
}

func TestForEachLogLineCarriageReturnNewlineIsOneBreak(t *testing.T) {
	got := readLogLines("hello\r\n\r\nworld\r\n")
	want := []string{"hello", "", "world"}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("lines = %#v, want %#v", got, want)
	}
}

func TestForEachLogLineSplitsLongLineOnRuneBoundary(t *testing.T) {
	// █ is three bytes, and 64KB is not a multiple of three, so a byte cap
	// lands inside a character. The two chunks must join back to the original.
	body := strings.Repeat("█", (maxLogLineBytes/3)+10)
	input := body + "\n"

	var chunks []string
	forEachLogLine(strings.NewReader(input), func(line string) {
		chunks = append(chunks, line)
	})
	if len(chunks) < 2 {
		t.Fatalf("expected the line to be split, got %d chunk(s)", len(chunks))
	}
	for i, chunk := range chunks {
		if !utf8.ValidString(chunk) {
			t.Fatalf("chunk %d is not valid UTF-8: %q", i, chunk[len(chunk)-4:])
		}
	}
	if strings.Join(chunks, "") != body {
		t.Fatalf("rejoined chunks changed the line: got %d bytes, want %d", len(strings.Join(chunks, "")), len(body))
	}
}

func readLogLines(input string) []string {
	var lines []string
	forEachLogLine(strings.NewReader(input), func(line string) {
		lines = append(lines, line)
	})
	return lines
}
