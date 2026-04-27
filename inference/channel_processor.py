import cv2
import numpy as np
from typing import List, Tuple


class ChannelProcessor:
    """Build piecewise channel combinations for the RGB PolyEdge signal."""

    @staticmethod
    def simulate_channel_process(
        r_range: Tuple[float, float],
        g_range: Tuple[float, float],
        b_range: Tuple[float, float],
    ) -> Tuple[List[str], List[List[float]]]:
        """Return active channel combinations and their normalized time windows."""
        points = sorted(
            set(
                [
                    0.0,
                    1.0,
                    r_range[0],
                    r_range[1],
                    g_range[0],
                    g_range[1],
                    b_range[0],
                    b_range[1],
                ]
            )
        )

        channel_list: List[str] = []
        channel_start_end_list: List[List[float]] = [[], []]
        prev_point = points[0]

        for current_point in points[1:]:
            active_channels: List[str] = []
            if r_range[0] <= prev_point < r_range[1]:
                active_channels.append("r")
            if g_range[0] <= prev_point < g_range[1]:
                active_channels.append("g")
            if b_range[0] <= prev_point < b_range[1]:
                active_channels.append("b")

            if not active_channels:
                active_channels.append("none")

            channel_list.append("".join(sorted(active_channels)))
            channel_start_end_list[0].append(prev_point)
            channel_start_end_list[1].append(current_point)
            prev_point = current_point

        return channel_list, channel_start_end_list

    @staticmethod
    def process_image_channels(
        rgb_image: np.ndarray,
        r_range: Tuple[float, float],
        g_range: Tuple[float, float],
        b_range: Tuple[float, float],
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """Return merged control images and the corresponding time windows."""
        r, g, b = cv2.split(rgb_image)
        channel_map = {"r": r, "g": g, "b": b}

        channel_list_str, channel_start_end_list = ChannelProcessor.simulate_channel_process(
            r_range,
            g_range,
            b_range,
        )

        channel_list_img: List[np.ndarray] = []
        for channels in channel_list_str:
            if channels == "none":
                combined = np.zeros_like(r)
            else:
                combined = np.zeros_like(r, dtype=np.float32)
                for ch in channels:
                    combined += channel_map[ch].astype(np.float32)
                combined = np.clip(combined, 0, 255).astype(np.uint8)

            channel_list_img.append(combined)

        return channel_list_img, channel_start_end_list
