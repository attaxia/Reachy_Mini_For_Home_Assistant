import unittest
from collections import deque

import numpy as np

from reachy_mini_hass.voice_assistant import AUDIO_BLOCK_SIZE, assemble_audio_chunk


class AssembleAudioChunkTests(unittest.TestCase):
    def test_exact_single_block(self):
        buf = deque([np.arange(AUDIO_BLOCK_SIZE, dtype=np.float32)])
        chunk = assemble_audio_chunk(buf, AUDIO_BLOCK_SIZE)

        self.assertEqual(len(chunk), AUDIO_BLOCK_SIZE)
        self.assertEqual(len(buf), 0)
        self.assertEqual(chunk[0], 0.0)
        self.assertEqual(chunk[-1], AUDIO_BLOCK_SIZE - 1)

    def test_spans_multiple_blocks_and_preserves_remainder(self):
        buf = deque(
            [
                np.arange(300, dtype=np.float32),
                np.arange(300, 600, dtype=np.float32),
            ]
        )
        chunk = assemble_audio_chunk(buf, 512)

        self.assertEqual(len(chunk), 512)
        # Samples must be gapless and in order
        np.testing.assert_array_equal(chunk, np.arange(512, dtype=np.float32))
        # Remainder of the second block goes back to the front
        self.assertEqual(len(buf), 1)
        np.testing.assert_array_equal(buf[0], np.arange(512, 600, dtype=np.float32))

    def test_consecutive_chunks_are_gapless(self):
        buf = deque([np.arange(1200, dtype=np.float32)])

        first = assemble_audio_chunk(buf, 512)
        second = assemble_audio_chunk(buf, 512)

        np.testing.assert_array_equal(first, np.arange(512, dtype=np.float32))
        np.testing.assert_array_equal(second, np.arange(512, 1024, dtype=np.float32))
        self.assertEqual(len(buf[0]), 1200 - 1024)


if __name__ == "__main__":
    unittest.main()
