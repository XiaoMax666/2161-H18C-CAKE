from pathlib import Path

from adapters.device_a_adapter import DeviceAAdapter


def main() -> None:
    data_dir = Path(r"C:\Users\new\Desktop\test_data\test_data\2025_04_12\001")

    files = {
        "GP3HD_data.csv": data_dir / "GP3HD_data.csv",
        "acc_marker_external.csv": data_dir / "acc_marker_external.csv",
        "Recording_info.csv": data_dir / "Recording_info.csv",
    }

    adapter = DeviceAAdapter(files)
    result = adapter.run()

    print("Upload success")
    print("Gaze rows:", len(result["gaze"]))
    print("Marker rows:", len(result["markers"]))

    print("\nMetadata:")
    print(result["metadata"])

    print("\nNormalized Gaze Data:")
    print(result["gaze"].head())

    print("\nNormalized Marker Data:")
    print(result["markers"].head())


if __name__ == "__main__":
    main()