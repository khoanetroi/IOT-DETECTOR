"""
run_test.py
Test the trained model by identifying IoT devices from CSV files.
Includes anomaly detection — unknown/malware traffic will be flagged.

Usage:
    # Test 1 file:
    python run_test.py --csv flows/SamsungCamera_d49a20d58cf4_flows.csv

    # Test all files in flows/:
    python run_test.py --all

    # Test a custom directory (e.g. demo_data from friend):
    python run_test.py --test_dir demo_data/known
    python run_test.py --test_dir demo_data/unknown --expect_unknown
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.inference import IoTFingerprinter


def test_single(fp, csv_path, expect_unknown=False):
    """Test a single CSV file."""
    filename = os.path.basename(csv_path)
    # Handle both formats: "AmazonEcho_xxx_flows.csv" and "test_AmazonEcho.csv"
    name_no_ext = filename.replace(".csv", "")
    if name_no_ext.startswith("test_"):
        true_device = name_no_ext[5:]  # strip "test_"
    else:
        true_device = name_no_ext.split("_")[0]

    try:
        result = fp.predict_from_csv(csv_path)
        predicted = result["predicted_device"]
        confidence = result["average_confidence"]
        distance = result.get("average_distance", -1)
        status = result.get("status", "")
        is_unknown = result.get("is_unknown", False)

        if expect_unknown:
            correct = is_unknown
            mark = "✅" if correct else "❌"
        else:
            correct = predicted == true_device
            mark = "✅" if correct else "❌"

        print(f"  {mark} File: {filename}")
        print(f"     Thuc te:    {true_device}")
        print(f"     AI du doan: {predicted} (conf: {confidence:.1%}, dist: {distance:.4f})")
        print(f"     Trang thai: {status}")
        print()
        return correct
    except Exception as e:
        print(f"  !! File: {filename} -> Loi: {e}\n")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test IoT Device Identification")
    parser.add_argument("--csv", type=str, help="Path to a single CSV file to test")
    parser.add_argument("--all", action="store_true", help="Test all CSV files in flows/")
    parser.add_argument("--test_dir", type=str, help="Test all CSVs in a custom directory")
    parser.add_argument("--expect_unknown", action="store_true",
                        help="Expect files to be unknown devices (anomaly detection test)")
    parser.add_argument("--checkpoint", type=str, default="experiments/finetune_best.pth")
    args = parser.parse_args()

    if not args.csv and not args.all and not args.test_dir:
        print("Usage:")
        print("  python run_test.py --csv flows/SamsungCamera_xxx.csv")
        print("  python run_test.py --all")
        print("  python run_test.py --test_dir demo_data/known")
        print("  python run_test.py --test_dir demo_data/unknown --expect_unknown")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  IoT Device Identification - TEST")
    print(f"{'='*60}\n")

    print("[1] Loading model...\n")
    fp = IoTFingerprinter(args.checkpoint)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}\n")

    if args.csv:
        test_single(fp, args.csv, args.expect_unknown)
        return

    if args.test_dir:
        csv_files = sorted(glob.glob(os.path.join(args.test_dir, "*.csv")))
    else:
        csv_files = sorted(glob.glob("flows/*.csv"))

    if not csv_files:
        print(f"[ERROR] No CSV files found!")
        sys.exit(1)

    print(f"  Found {len(csv_files)} CSV files")
    if args.expect_unknown:
        print(f"  Mode: Expecting UNKNOWN devices (anomaly detection test)\n")
    else:
        print()

    correct = 0
    total = 0
    for csv_path in csv_files:
        result = test_single(fp, csv_path, args.expect_unknown)
        if result is not None:
            total += 1
            if result:
                correct += 1

    acc = correct / total if total > 0 else 0
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"  Correct: {correct}/{total} ({acc:.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
