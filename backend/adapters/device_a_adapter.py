import pandas as pd
from typing import Dict

from adapters.base_adapter import BaseAdapter, InvalidFormatError


class DeviceAAdapter(BaseAdapter):
    """
    Adapter for Device A (GP3 dataset).
    """

    required_files = (
        "GP3HD_data.csv",
        "acc_marker_external.csv",
        "Recording_info.csv",
    )

    def load_raw_data(self) -> None:
        self.raw_data["gaze"] = self.read_csv("GP3HD_data.csv")
        self.raw_data["markers"] = self.read_csv("acc_marker_external.csv", header=None)
        self.raw_data["info"] = self.read_csv("Recording_info.csv", header=None)

    def validate_content(self) -> None:
        gaze = self.raw_data["gaze"]
        markers = self.raw_data["markers"]
        info = self.raw_data["info"]

        self.require_columns(
            gaze,
            ["Time_s", "FPOGX", "FPOGY", "FPOGV"],
            "GP3HD_data.csv",
        )

        if markers.shape[1] < 2:
            raise InvalidFormatError(
                "acc_marker_external.csv must contain at least two columns "
                "(start_time, end_time)."
            )

        if info.shape[1] < 2:
            raise InvalidFormatError(
                "Recording_info.csv must contain at least two columns (key, value)."
            )

        info_keys = set(info.iloc[:, 0].astype(str).str.strip())

        if "width" not in info_keys or "height" not in info_keys:
            raise InvalidFormatError(
                "Recording_info.csv must contain 'width' and 'height' rows."
            )

        # validate gaze attributes
        for col in ["Time_s", "FPOGX", "FPOGY", "FPOGV"]:
            try:
                pd.to_numeric(gaze[col])
            except Exception as e:
                raise InvalidFormatError(
                    "Column '{}' in GP3HD_data.csv contains non-numeric values.".format(col)
                ) from e

        # Validate marker timestamp
        try:
            start = pd.to_numeric(markers.iloc[:, 0])
            end = pd.to_numeric(markers.iloc[:, 1])
        except Exception as e:
            raise InvalidFormatError(
                "acc_marker_external.csv contains non-numeric marker timestamps."
            ) from e

        if (end < start).any():
            raise InvalidFormatError(
                "acc_marker_external.csv contains rows where end_time < start_time."
            )

    def _get_info_value(self, key: str) -> str:
        """
        Extract a value from Recording_info.csv key-value rows.
        """
        info = self.raw_data["info"]
        matched = info[info.iloc[:, 0].astype(str).str.strip() == key]

        if matched.empty:
            raise InvalidFormatError(
                "Recording_info.csv is missing required key '{}'.".format(key)
            )

        return str(matched.iloc[0, 1]).strip()

    def normalize(self) -> Dict[str, pd.DataFrame]:
        gaze = self.raw_data["gaze"].copy()
        markers = self.raw_data["markers"].copy()

        try:
            screen_width = float(self._get_info_value("width"))
            screen_height = float(self._get_info_value("height"))
        except Exception as e:
            raise InvalidFormatError(
                "Failed to parse width/height from Recording_info.csv."
            ) from e

        gaze["Time_s"] = pd.to_numeric(gaze["Time_s"])
        gaze["FPOGX"] = pd.to_numeric(gaze["FPOGX"])
        gaze["FPOGY"] = pd.to_numeric(gaze["FPOGY"])
        gaze["FPOGV"] = pd.to_numeric(gaze["FPOGV"])

        markers.iloc[:, 0] = pd.to_numeric(markers.iloc[:, 0])
        markers.iloc[:, 1] = pd.to_numeric(markers.iloc[:, 1])

        # Initialize the marker starting timestamp as 0
        time_zero = markers.iloc[:, 0].min()

        gaze_df = pd.DataFrame({
            "timestamp": (gaze["Time_s"] - time_zero).round(6),
            "gaze_x_norm": gaze["FPOGX"],
            "gaze_y_norm": gaze["FPOGY"],
            "valid": gaze["FPOGV"],
            "device": "DeviceA",
        })

        gaze_df["gaze_x_px"] = (gaze_df["gaze_x_norm"] * screen_width).round(2)
        gaze_df["gaze_y_px"] = (gaze_df["gaze_y_norm"] * screen_height).round(2)

        marker_df = pd.DataFrame({
            "start_time": (markers.iloc[:, 0] - time_zero).round(6),
            "end_time": (markers.iloc[:, 1] - time_zero).round(6),
        })

        metadata_df = pd.DataFrame({
            "screen_width": [screen_width],
            "screen_height": [screen_height],
            "device": ["DeviceA"],
            "time_zero_source": ["first_marker_start"],
        })

        return {
            "gaze": gaze_df,
            "markers": marker_df,
            "metadata": metadata_df,
        }