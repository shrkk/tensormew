"""Real-emulator regression: fresh pixels without extra game frames or RAM changes."""
import numpy as np
import pytest

from src.config import ROM_PATH, SYM_PATH, STATES_DIR
from src.emulator import Emulator
from src.states import StateRegistry


@pytest.mark.skipif(not (ROM_PATH.exists() and SYM_PATH.exists() and
                       (STATES_DIR / 'new_bark_outside.state').exists()), reason='ROM/start fixture unavailable')
def test_headless_pixels_refresh_with_exact_public_tick_reference():
    fixed = Emulator(headless=True, speed=0)
    reference = Emulator(headless=True, speed=0)
    try:
        saved = StateRegistry().load_bytes('new_bark_outside')
        fixed.load_state(saved, apply_options=False)
        reference.load_state(saved, apply_options=False)
        first_pixels = np.asarray(fixed.screen_image()).copy()
        native_start = fixed.pyboy.frame_count
        ref_start = reference.pyboy.frame_count
        elapsed = 0
        changed = False
        for button in ('left', 'left', 'down', 'right'):
            fixed.hold(button)
            reference.hold(button)
            fixed.tick(8, render=False)
            reference.pyboy.tick(8, False)
            fixed.release(button)
            reference.release(button)
            fixed.tick(12, render=False)
            reference.pyboy.tick(12, True)
            elapsed += 20
            assert fixed.frame == elapsed
            assert fixed.pyboy.frame_count - native_start == elapsed
            assert reference.pyboy.frame_count - ref_start == elapsed
            # Includes map/coordinates, battle/menu state and game RNG, not just a visual assertion.
            assert fixed.read(0, 0xC000, 0x1000) == reference.read(0, 0xC000, 0x1000)
            for bank in range(1, 8):
                assert fixed.read(bank, 0xD000, 0x1000) == reference.read(bank, 0xD000, 0x1000)
            assert fixed.read(0, 0xFF80, 0x7F) == reference.read(0, 0xFF80, 0x7F)
            pixels = np.asarray(fixed.screen_image())
            np.testing.assert_array_equal(pixels, np.asarray(reference.screen_image()))
            changed |= not np.array_equal(first_pixels, pixels)
        assert changed, 'Pixels must follow real game actions instead of remaining at the loaded frame'
    finally:
        fixed.close()
        reference.close()
