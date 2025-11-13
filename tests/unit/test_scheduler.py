"""Unit tests for EventScheduler."""

import pytest

from seapopym_message.distributed.scheduler import EventScheduler


@pytest.mark.unit
class TestEventScheduler:
    """Test EventScheduler priority queue and scheduling logic."""

    def test_scheduler_creation(self) -> None:
        """Test creating a scheduler with mock workers."""
        # Mock workers (empty list for unit test)
        workers = []

        scheduler = EventScheduler(workers=workers, dt=0.1, t_max=10.0)

        assert scheduler.dt == 0.1
        assert scheduler.t_max == 10.0
        assert scheduler.t_current == 0.0
        assert len(scheduler.workers) == 0

    def test_schedule_and_pop_events(self) -> None:
        """Test priority queue operations."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=10.0)

        # Schedule events out of order
        scheduler._schedule_event(time=1.0, event_type="output", data={"key": "val1"})
        scheduler._schedule_event(time=0.5, event_type="step", data={"key": "val2"})
        scheduler._schedule_event(time=2.0, event_type="step", data={"key": "val3"})

        # Events should be popped in time order
        event1 = scheduler._pop_event()
        assert event1 is not None
        assert event1[0] == 0.0  # Initial step event scheduled in __init__
        assert event1[1] == "step"

        event2 = scheduler._pop_event()
        assert event2 is not None
        assert event2[0] == 0.5
        assert event2[1] == "step"
        assert event2[2]["key"] == "val2"

        event3 = scheduler._pop_event()
        assert event3 is not None
        assert event3[0] == 1.0
        assert event3[1] == "output"

        event4 = scheduler._pop_event()
        assert event4 is not None
        assert event4[0] == 2.0

    def test_pop_empty_queue(self) -> None:
        """Test popping from empty queue."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=10.0)

        # Pop initial event
        scheduler._pop_event()

        # Queue should now be empty
        event = scheduler._pop_event()
        assert event is None

    def test_aggregate_diagnostics_empty(self) -> None:
        """Test aggregating empty diagnostics."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=1.0)

        diagnostics = scheduler._aggregate_diagnostics([])

        assert diagnostics["t"] == 0.0
        assert diagnostics["num_workers"] == 0
        assert diagnostics["diagnostics"] == []

    def test_aggregate_diagnostics_single_worker(self) -> None:
        """Test aggregating diagnostics from single worker."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=1.0)
        scheduler.t_current = 0.5

        worker_diag = [{"worker_id": 0, "t": 0.5, "biomass_mean": 42.0}]

        aggregated = scheduler._aggregate_diagnostics(worker_diag)

        assert aggregated["t"] == 0.5
        assert aggregated["num_workers"] == 1
        assert aggregated["biomass_global_mean"] == 42.0
        assert aggregated["biomass_global_min"] == 42.0
        assert aggregated["biomass_global_max"] == 42.0

    def test_aggregate_diagnostics_multiple_workers(self) -> None:
        """Test aggregating diagnostics from multiple workers."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=1.0)
        scheduler.t_current = 1.0

        worker_diags = [
            {"worker_id": 0, "t": 1.0, "biomass_mean": 10.0},
            {"worker_id": 1, "t": 1.0, "biomass_mean": 20.0},
            {"worker_id": 2, "t": 1.0, "biomass_mean": 30.0},
        ]

        aggregated = scheduler._aggregate_diagnostics(worker_diags)

        assert aggregated["t"] == 1.0
        assert aggregated["num_workers"] == 3
        assert aggregated["biomass_global_mean"] == 20.0  # (10+20+30)/3
        assert aggregated["biomass_global_min"] == 10.0
        assert aggregated["biomass_global_max"] == 30.0

    def test_aggregate_multiple_state_variables(self) -> None:
        """Test aggregating multiple state variables."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=1.0)

        worker_diags = [
            {"worker_id": 0, "t": 0.1, "biomass_mean": 10.0, "temperature_mean": 15.0},
            {"worker_id": 1, "t": 0.1, "biomass_mean": 20.0, "temperature_mean": 25.0},
        ]

        aggregated = scheduler._aggregate_diagnostics(worker_diags)

        # Check biomass
        assert aggregated["biomass_global_mean"] == 15.0
        assert aggregated["biomass_global_min"] == 10.0
        assert aggregated["biomass_global_max"] == 20.0

        # Check temperature
        assert aggregated["temperature_global_mean"] == 20.0
        assert aggregated["temperature_global_min"] == 15.0
        assert aggregated["temperature_global_max"] == 25.0

    def test_get_current_time(self) -> None:
        """Test getting current simulation time."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=10.0)

        assert scheduler.get_current_time() == 0.0

        scheduler.t_current = 5.5
        assert scheduler.get_current_time() == 5.5

    def test_repr(self) -> None:
        """Test string representation."""
        scheduler = EventScheduler(workers=[], dt=0.1, t_max=10.0)

        repr_str = repr(scheduler)
        assert "EventScheduler" in repr_str
        assert "dt=0.1" in repr_str
        assert "t_max=10.0" in repr_str
        assert "t_current=0.00" in repr_str
