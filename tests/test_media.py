from fractions import Fraction
from types import SimpleNamespace

import av
import pytest

from body_eye_sync.media import container_origin, stream_duration, stream_start


def _stream(start, duration=None):
    return SimpleNamespace(
        start_time=start,
        duration=duration,
        time_base=Fraction(1, 1),
    )


def test_streams_share_the_earliest_stream_origin_when_container_has_none():
    first = _stream(100)
    second = _stream(101)
    container = SimpleNamespace(start_time=None, streams=[first, second])

    assert container_origin(container) == pytest.approx(100.0)
    assert stream_start(container, first) == pytest.approx(0.0)
    assert stream_start(container, second) == pytest.approx(1.0)


def test_stream_duration_fallback_excludes_its_delayed_start():
    first = _stream(0)
    delayed = _stream(1)
    container = SimpleNamespace(
        start_time=0,
        duration=3 * av.time_base,
        streams=[first, delayed],
    )

    assert stream_duration(container, delayed) == pytest.approx(2.0)
