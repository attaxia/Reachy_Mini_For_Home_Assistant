"""SDK 1.9 shares one GStreamer pipeline between mic capture and playback:
media.stop_playing() sets it to NULL, silently killing the microphone (no
wake words, no STT audio). These tests pin the app-side workaround: every
mid-session playback stop must re-arm capture via start_recording().
"""

import types
import unittest
from pathlib import Path

from reachy_mini_hass.audio.audio_player_shared import stop_playback_keep_mic


class StopPlaybackKeepMicTests(unittest.TestCase):
    def test_stops_playback_then_rearms_recording(self):
        calls = []
        media = types.SimpleNamespace(
            stop_playing=lambda: calls.append("stop_playing"),
            start_recording=lambda: calls.append("start_recording"),
        )

        stop_playback_keep_mic(types.SimpleNamespace(media=media))

        self.assertEqual(calls, ["stop_playing", "start_recording"])

    def test_recording_rearm_failure_does_not_raise(self):
        def boom():
            raise RuntimeError("recording unavailable")

        media = types.SimpleNamespace(stop_playing=lambda: None, start_recording=boom)

        stop_playback_keep_mic(types.SimpleNamespace(media=media))  # must not raise


class NoDirectStopPlayingInPlayersTests(unittest.TestCase):
    """Player code must never call media.stop_playing() directly — that is
    reserved for shutdown/suspend paths where the mic is meant to stop."""

    PLAYER_FILES = [
        "audio_player_playback.py",
        "audio_player_local.py",
        "audio_player_sendspin.py",
        "audio_player.py",
        "local_audio_player.py",
        "audio_player_stream_decoded.py",
        "audio_player_stream_pcm.py",
    ]

    def test_players_use_mic_preserving_stop(self):
        for fname in self.PLAYER_FILES:
            path = Path("reachy_mini_hass/audio") / fname
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("media.stop_playing", content, f"{fname} calls media.stop_playing() directly")


if __name__ == "__main__":
    unittest.main()
