# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller manager for registering and managing control loops."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from threading import RLock, local
from typing import Self

from .controller import Loop

logger = getLogger(__name__)

_MISSING = object()


def _has_attr(obj: object, name: str) -> bool:
    """Check attribute presence without risking a property's internal AttributeError being read as "missing"."""
    if name in getattr(obj, "__dict__", {}):
        return True
    return getattr(type(obj), name, _MISSING) is not _MISSING


class ControllerLifecycleState(str, Enum):
    """Runner-visible lifecycle state for a controller or required sidecar."""

    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"


@dataclass
class _ControllerRecord:
    generation: int
    state: ControllerLifecycleState = ControllerLifecycleState.STARTING
    loop_names: set[str] = field(default_factory=set)
    ever_registered: bool = False
    failure_reason: str | None = None


class ControllerManager:
    """Singleton manager for registering control loops and validating their health."""

    _instance: Self | None = None
    _instance_lock = RLock()

    def __init__(self) -> None:
        """Private constructor. Use :meth:`get_instance` instead."""
        if ControllerManager._instance is not None:
            raise RuntimeError("Use ControllerManager.get_instance() to get the singleton instance")
        self._init_state()

    def _init_state(self) -> None:
        """Initialize mutable singleton state."""
        self._loops: dict[str, Loop] = {}
        self._reported_health: dict[str, bool] = {}
        self._controllers: dict[str, _ControllerRecord] = {}
        self._controller_generations: dict[str, int] = {}
        self._loop_controllers: dict[str, tuple[str, int]] = {}
        self._controller_context = local()
        self._lock = RLock()

    @classmethod
    def get_instance(cls) -> "ControllerManager":
        """Get the process-wide controller manager."""
        with cls._instance_lock:
            if cls._instance is None:
                instance = cls.__new__(cls)
                instance._init_state()
                cls._instance = instance
        return cls._instance

    def await_controller_registration(self, name: str) -> int:
        """Start a lifecycle generation that must register at least one loop.

        Calls made from the active registration context are idempotent. This is
        useful for components that can run both through the platform runner and
        directly in tests or standalone processes.
        """
        with self._lock:
            context = getattr(self._controller_context, "value", None)
            if context is not None and context[0] == name:
                record = self._controllers.get(name)
                if record is not None and record.generation == context[1]:
                    return record.generation

            existing = self._controllers.get(name)
            if existing is not None and existing.state is ControllerLifecycleState.STOPPING:
                raise RuntimeError(f"Controller '{name}' is still stopping")
            if existing is not None:
                if existing.state is not ControllerLifecycleState.FAILED:
                    raise RuntimeError(f"Controller '{name}' is already {existing.state.value}")
                live_loops = self._live_loop_names_locked(existing.loop_names)
                if live_loops:
                    raise RuntimeError(
                        f"Controller '{name}' still has running loop(s): {', '.join(sorted(live_loops))}"
                    )
                self._remove_controller_loops_locked(name, existing)

            generation = self._controller_generations.get(name, 0) + 1
            self._controller_generations[name] = generation
            self._controllers[name] = _ControllerRecord(generation=generation)
            self._reported_health.pop(name, None)
            return generation

    def mark_controller_failed(
        self,
        name: str,
        generation: int | None = None,
        *,
        reason: str | None = None,
    ) -> bool:
        """Mark the current lifecycle generation failed.

        Returns ``False`` when a delayed callback refers to an older generation.

        A no-op if the generation is already :attr:`ControllerLifecycleState.STOPPING`:
        that state means a resource this generation still owns (e.g. a listening
        socket) has not been released yet, and a *later*, more generic failure
        report for the same generation (for example the runner's catch-all
        "controller crashed" handler, invoked after the component's own code
        already transitioned it to STOPPING and re-raised) must not downgrade it
        back to FAILED — doing so would drop the "still stopping" guard in
        :meth:`await_controller_registration` and let a new generation start
        while the old one still holds the resource.
        """
        with self._lock:
            record = self._record_for_update_locked(name, generation)
            if record is None:
                if generation is not None:
                    return False
                generation = self._controller_generations.get(name, 0) + 1
                self._controller_generations[name] = generation
                record = _ControllerRecord(generation=generation)
                self._controllers[name] = record
            if record.state is ControllerLifecycleState.STOPPING:
                return True
            record.state = ControllerLifecycleState.FAILED
            record.failure_reason = reason
            return True

    def mark_controller_stopping(self, name: str, generation: int) -> bool:
        """Transition an over-time generation to ``STOPPING``.

        Returns ``True`` only for the caller that performs the transition. A
        component that already entered ``STOPPING`` owns its delayed-resource
        cleanup, so generic runner cleanup must not install a competing watcher.
        """
        with self._lock:
            record = self._record_for_update_locked(name, generation)
            if record is None:
                return False
            if record.state is ControllerLifecycleState.STOPPING:
                return False
            record.state = ControllerLifecycleState.STOPPING
            if record.failure_reason is None:
                record.failure_reason = "controller thread did not stop before the shutdown deadline"
            return True

    def stop_tracking_controller(
        self,
        name: str,
        generation: int | None = None,
        *,
        clear_state: bool = True,
        allow_stopping: bool = False,
    ) -> bool:
        """Remove stopped loops and optionally the lifecycle state.

        Generation matching prevents delayed cleanup from erasing a newer run.
        ``clear_state=False`` removes stale loop objects while preserving a
        startup failure.
        """
        with self._lock:
            record = self._record_for_update_locked(name, generation)
            if record is None:
                return False
            if record.state is ControllerLifecycleState.STOPPING and not allow_stopping:
                return False
            still_running = self._live_loop_names_locked(record.loop_names)
            if still_running:
                logger.warning(
                    "Leaving health tracking in place for %r; loop(s) %s still running",
                    name,
                    sorted(still_running),
                )
                return False

            self._remove_controller_loops_locked(name, record)
            if clear_state:
                self._controllers.pop(name, None)
                self._reported_health.pop(name, None)
            else:
                record.state = ControllerLifecycleState.FAILED
            return True

    def _record_for_update_locked(self, name: str, generation: int | None) -> _ControllerRecord | None:
        record = self._controllers.get(name)
        if record is None:
            return None
        if generation is not None and record.generation != generation:
            return None
        return record

    def _remove_controller_loops_locked(self, name: str, record: _ControllerRecord) -> None:
        for loop_name in set(record.loop_names):
            owner = self._loop_controllers.get(loop_name)
            if owner == (name, record.generation):
                self._loop_controllers.pop(loop_name, None)
                self._loops.pop(loop_name, None)
                self._reported_health.pop(loop_name, None)
        record.loop_names.clear()

    def _live_loop_names_locked(self, loop_names: set[str]) -> set[str]:
        """Return which of ``loop_names`` are registered threads that are still alive."""
        return {
            loop_name
            for loop_name in loop_names
            if isinstance((loop := self._loops.get(loop_name)), threading.Thread) and loop.is_alive()
        }

    def watch_delayed_exit(
        self,
        thread: threading.Thread,
        name: str,
        generation: int,
        *,
        clear_state: bool = True,
        thread_name: str | None = None,
    ) -> None:
        """Untrack ``name``'s lifecycle generation, in the background, once ``thread`` exits.

        Used after a shutdown deadline passes while ``thread`` is still alive: the
        caller has already called :meth:`mark_controller_stopping` to keep the
        generation reported unhealthy, and this spawns the watcher that clears it
        once the thread actually finishes (or, with ``clear_state=False``, leaves
        the failure recorded while dropping the now-dead loop object).
        """

        def _wait_and_untrack() -> None:
            thread.join()
            if self.stop_tracking_controller(name, generation, clear_state=clear_state, allow_stopping=True):
                return
            if self.watch_controller_loops_exit(name, generation, clear_state=clear_state):
                return
            # A loop may have exited between the failed untrack attempt and
            # the watcher snapshot. Retry once with the same generation token.
            self.stop_tracking_controller(name, generation, clear_state=clear_state, allow_stopping=True)

        threading.Thread(
            target=_wait_and_untrack,
            name=thread_name or f"{thread.name}-cleanup",
            daemon=True,
        ).start()

    def watch_controller_loops_exit(
        self,
        name: str,
        generation: int,
        *,
        clear_state: bool = True,
    ) -> bool:
        """Untrack a generation after all of its currently live loops exit.

        Returns ``False`` when the generation is stale or has no live loop
        threads to watch. The generation check in the eventual cleanup keeps a
        delayed watcher from erasing a later restart.
        """
        with self._lock:
            record = self._record_for_update_locked(name, generation)
            if record is None:
                return False
            live_loops = [
                loop
                for loop_name in record.loop_names
                if isinstance((loop := self._loops.get(loop_name)), threading.Thread) and loop.is_alive()
            ]
        if not live_loops:
            return False

        def _wait_and_untrack() -> None:
            for loop in live_loops:
                loop.join()
            self.stop_tracking_controller(name, generation, clear_state=clear_state, allow_stopping=True)

        threading.Thread(
            target=_wait_and_untrack,
            name=f"controller-{name}-loops-cleanup",
            daemon=True,
        ).start()
        return True

    @contextmanager
    def controller_registration_context(self, name: str, generation: int | None = None) -> Iterator[None]:
        """Associate loop registrations in the current thread with a generation."""
        previous = getattr(self._controller_context, "value", None)
        if generation is None:
            with self._lock:
                record = self._controllers.get(name)
                generation = record.generation if record is not None else self.await_controller_registration(name)
        self._controller_context.value = (name, generation)
        try:
            yield
        finally:
            self._controller_context.value = previous

    def register(self, name: str, loop: Loop) -> None:
        """Register a control loop with a unique name."""
        with self._lock:
            if name in self._loops:
                raise ValueError(f"Loop with name '{name}' is already registered")
            context = getattr(self._controller_context, "value", None)
            if context is not None:
                controller_name, generation = context
                record = self._controllers.get(controller_name)
                if generation is None or record is None or record.generation != generation:
                    raise RuntimeError(f"Controller '{controller_name}' registration belongs to a stale generation")

            loop.name = name
            self._loops[name] = loop
            if context is not None:
                record.loop_names.add(name)
                record.ever_registered = True
                record.state = ControllerLifecycleState.RUNNING
                record.failure_reason = None
                self._loop_controllers[name] = (controller_name, generation)
        logger.debug("Registered loop: %s", name)

    def unregister(self, name: str) -> None:
        """Unregister a loop by name."""
        with self._lock:
            if name not in self._loops:
                raise KeyError(f"No loop with name '{name}' is registered")
            del self._loops[name]
            self._reported_health.pop(name, None)
            owner = self._loop_controllers.pop(name, None)
            if owner is not None:
                controller_name, generation = owner
                record = self._controllers.get(controller_name)
                if record is not None and record.generation == generation:
                    record.loop_names.discard(name)
                    if not record.loop_names and record.state is ControllerLifecycleState.RUNNING:
                        record.state = ControllerLifecycleState.FAILED
                        record.failure_reason = "controller unregistered its last control loop"
        logger.info("Unregistered loop: %s", name)

    def get_loop(self, name: str) -> Loop:
        """Get a registered loop by name."""
        with self._lock:
            if name not in self._loops:
                raise KeyError(f"No loop with name '{name}' is registered")
            return self._loops[name]

    def get_all_loops(self) -> dict[str, Loop]:
        """Get a snapshot of all registered loops."""
        with self._lock:
            return self._loops.copy()

    def validate_all_healthy(self, detailed: bool = True) -> tuple[bool, dict[str, bool]]:
        """Validate runner lifecycle state and every registered loop."""
        with self._lock:
            loops = self._loops.copy()
            unhealthy_controllers = {
                name: (record.state, record.ever_registered, record.failure_reason)
                for name, record in self._controllers.items()
                if record.state is not ControllerLifecycleState.RUNNING
            }

        if not loops and not unhealthy_controllers:
            logger.debug("No loops registered for health validation")
            return True, {}

        health_status = {name: False for name in sorted(unhealthy_controllers)} if detailed else {}
        all_healthy = not unhealthy_controllers

        for name, (state, ever_registered, failure_reason) in unhealthy_controllers.items():
            reason = failure_reason
            if reason is None:
                if state is ControllerLifecycleState.STARTING:
                    reason = "controller has not registered a control loop"
                elif state is ControllerLifecycleState.STOPPING:
                    reason = "controller is still stopping"
                elif ever_registered:
                    reason = "controller exited unexpectedly after registering a control loop"
                else:
                    reason = "controller failed before registering a control loop"
            self._log_health_transition(name, False, reason)

        for name, loop in loops.items():
            if name in unhealthy_controllers:
                continue
            try:
                is_healthy = loop.is_healthy if _has_attr(loop, "is_healthy") else True
                if detailed:
                    health_status[name] = is_healthy
                if not is_healthy:
                    all_healthy = False
                    logger.debug("Loop '%s' is unhealthy", name)
                unhealthy_reason = loop.unhealthy_reason if _has_attr(loop, "unhealthy_reason") else None
                self._log_health_transition(name, is_healthy, unhealthy_reason)
            except Exception as error:
                if detailed:
                    health_status[name] = False
                all_healthy = False
                logger.error("Error checking health of loop '%s': %s", name, error, exc_info=True)
                self._log_health_transition(name, False, f"health check raised: {error}")

        return all_healthy, health_status if detailed else {}

    def _log_health_transition(self, name: str, is_healthy: bool, reason: str | None) -> None:
        """Log only when a loop's health changes."""
        with self._lock:
            previous = self._reported_health.get(name)
            if previous == is_healthy:
                return
            self._reported_health[name] = is_healthy

            if is_healthy:
                if previous is not None:
                    logger.info("Control loop '%s' is healthy again", name)
            else:
                logger.warning("Control loop '%s' is unhealthy: %s", name, reason or "reason unavailable")

    def clear(self) -> None:
        """Clear loops and lifecycle state without reusing generation tokens."""
        with self._lock:
            count = len(self._loops)
            self._loops.clear()
            self._reported_health.clear()
            self._controllers.clear()
            self._loop_controllers.clear()
        logger.info("Cleared %d registered loop(s)", count)
