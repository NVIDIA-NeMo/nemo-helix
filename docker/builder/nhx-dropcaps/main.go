// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

// nhx-dropcaps runs a program with no capabilities but the five kaniko needs, the plain pod sandbox's:
//
//	nhx-dropcaps PROGRAM [ARG...]
//
// An OpenSandbox sandbox can't get them from its pod: when the server adds its egress sidecar, it replaces the
// template's capability drop list with its own, so the sandbox keeps the container runtime's defaults. One of those,
// NET_RAW, lets a `RUN` mark its packets past the sidecar's firewall, or send frames around it. So this drops every
// other capability from the bounding, inheritable and ambient sets, sets no_new_privs, checks all of it, and only then
// runs the program. If any of that fails, it runs nothing and exits 125.
//
// It holds only while nothing the caller wrote has run in the sandbox: a `RUN` could replace this binary, or the shell
// that starts it. So the builder runs it once per sandbox, with each image in a sandbox of its own.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"unsafe"
)

// The capabilities kept: what kaniko needs to unpack layers whose files belong to many users.
const keep uint64 = 1<<0 | // CAP_CHOWN
	1<<1 | // CAP_DAC_OVERRIDE
	1<<3 | // CAP_FOWNER
	1<<6 | // CAP_SETGID
	1<<7 // CAP_SETUID

const (
	prCapbsetRead        = 23
	prCapbsetDrop        = 24
	prSetNoNewPrivs      = 38
	prGetNoNewPrivs      = 39
	prCapAmbient         = 47
	prCapAmbientIsSet    = 1
	prCapAmbientClearAll = 4
	linuxCapabilityV3    = 0x20080522
)

type capHeader struct {
	version uint32
	pid     int32
}

// One per 32 capabilities.
type capData struct {
	effective, permitted, inheritable uint32
}

func init() {
	// Capabilities are a thread's own: every call below, and the exec, must be on this one.
	runtime.LockOSThread()
}

func main() {
	if len(os.Args) < 2 {
		fail("usage: nhx-dropcaps PROGRAM [ARG...]")
	}
	program, err := exec.LookPath(os.Args[1])
	if err != nil {
		fail("%v", err)
	}

	// The bounding set caps what any later exec can hold, even as root.
	for c := uintptr(0); ; c++ {
		held, known := inBoundingSet(c)
		if !known {
			break
		}
		if held && !kept(c) {
			if _, err := prctl(prCapbsetDrop, c); err != nil {
				fail("dropping capability %d: %v", c, err)
			}
		}
	}
	// An exec as root adds the inheritable set to what the program holds.
	header := capHeader{version: linuxCapabilityV3}
	var data [2]capData
	if err := capget(&header, &data); err != nil {
		fail("reading capabilities: %v", err)
	}
	data[0].inheritable &= uint32(keep)
	data[1].inheritable &= uint32(keep >> 32)
	if err := capset(&header, &data); err != nil {
		fail("setting capabilities: %v", err)
	}
	// The ambient set is kept across an exec as it is.
	if _, err := prctl(prCapAmbient, prCapAmbientClearAll); err != nil {
		fail("clearing the ambient set: %v", err)
	}
	// No setuid binary or file capability can add any back.
	if _, err := prctl(prSetNoNewPrivs, 1); err != nil {
		fail("setting no_new_privs: %v", err)
	}

	check()
	err = syscall.Exec(program, os.Args[1:], os.Environ())
	fail("running %s: %v", program, err)
}

// check fails unless every set the program could get a capability from holds no more than it keeps.
func check() {
	for c := uintptr(0); ; c++ {
		held, known := inBoundingSet(c)
		if !known {
			break
		}
		if kept(c) {
			continue
		}
		if held {
			fail("capability %d is still in the bounding set", c)
		}
		if ambient, err := prctl(prCapAmbient, prCapAmbientIsSet, c); err != nil || ambient != 0 {
			fail("capability %d may still be in the ambient set", c)
		}
	}
	header := capHeader{version: linuxCapabilityV3}
	var data [2]capData
	if err := capget(&header, &data); err != nil {
		fail("reading capabilities: %v", err)
	}
	inheritable := uint64(data[0].inheritable) | uint64(data[1].inheritable)<<32
	if inheritable&^keep != 0 {
		fail("the inheritable set still holds more than it keeps")
	}
	if nnp, err := prctl(prGetNoNewPrivs); err != nil || nnp != 1 {
		fail("no_new_privs is not set")
	}
}

func kept(c uintptr) bool {
	return c < 64 && keep&(1<<c) != 0
}

// inBoundingSet reports whether capability c is in this thread's bounding set, and whether the kernel knows c at all.
func inBoundingSet(c uintptr) (held, known bool) {
	r, err := prctl(prCapbsetRead, c)
	if err != nil {
		return false, false
	}
	return r == 1, true
}

func prctl(option uintptr, args ...uintptr) (uintptr, error) {
	var a [4]uintptr
	copy(a[:], args)
	r, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, option, a[0], a[1], a[2], a[3], 0)
	if errno != 0 {
		return 0, errno
	}
	return r, nil
}

func capget(header *capHeader, data *[2]capData) error {
	_, _, errno := syscall.RawSyscall(
		syscall.SYS_CAPGET, uintptr(unsafe.Pointer(header)), uintptr(unsafe.Pointer(&data[0])), 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func capset(header *capHeader, data *[2]capData) error {
	_, _, errno := syscall.RawSyscall(
		syscall.SYS_CAPSET, uintptr(unsafe.Pointer(header)), uintptr(unsafe.Pointer(&data[0])), 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "nhx-dropcaps: "+format+"\n", args...)
	os.Exit(125)
}
