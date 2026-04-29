"""
run_test.py
Test the trained model by identifying IoT devices from CSV files.

Usage:
    # Test 1 file cụ thể:
    python run_test.py --csv flows/SamsungCamera_d49a20d58cf4_flows.csv

    # Test tất cả file trong thư mục flows/:
    python run_test.py --all

    # Test với model checkpoint khác:
    python run_test.py --all --checkpoint experiments/finetune_best.pth
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.inference import IoTFingerprinter


def test_single(fp, csv_path):
    """Test a single CSV file."""
    filename = os.path.basename(csv_path)
    # Tên thiết bị thật lấy từ tên file (VD: SamsungCamera_xxx.csv → SamsungCamera)
    true_device = filename.split("_")[0]

    try:
        result = fp.predict_from_csv(csv_path)
        predicted = result["predicted_device"]
        confidence = result["average_confidence"]
        correct = "✅" if predicted == true_device else "❌"

        print(f"  {correct} File: {filename}")
        print(f"     Thực tế:    {true_device}")
        print(f"     AI dự đoán: {predicted} (confidence: {confidence:.1%})")
        print(f"     Chi tiết:   {result['all_votes']}")
        print()
        return predicted == true_device
    except Exception as e:
        print(f"  ⚠️  File: {filename} -> Lỗi: {e}\n")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test IoT Device Identification")
    parser.add_argument("--csv", type=str, help="Path to a single CSV file to test")
    parser.add_argument("--all", action="store_true", help="Test all CSV files in flows/")
    parser.add_argument("--checkpoint", type=str, default="experiments/finetune_best.pth")
    args = parser.parse_args()

    if not args.csv and not args.all:
        print("Usage:")
        print("  python run_test.py --csv flows/SamsungCamera_xxx.csv")
        print("  python run_test.py --all")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  IoT Device Identification - TEST")
    print(f"{'='*60}\n")

    # Load model
    print("[1] Loading model...\n")
    fp = IoTFingerprinter(args.checkpoint)

    print(f"\n{'─'*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}\n")

    if args.csv:
        # Test 1 file
        test_single(fp, args.csv)

    elif args.all:
        # Test tất cả file CSV
        csv_files = sorted(glob.glob("flows/*.csv"))
        if not csv_files:
            print("[ERROR] Khong tim thay file CSV trong thu muc flows/")
            sys.exit(1)

        print(f"  Found {len(csv_files)} CSV files\n")

        correct = 0
        total = 0
        for csv_path in csv_files:
            result = test_single(fp, csv_path)
            if result is not None:
                total += 1
                if result:
                    correct += 1

        # Summary
        acc = correct / total if total > 0 else 0
        print(f"{'='*60}")
        print(f"  SUMMARY")
        print(f"  Correct: {correct}/{total} ({acc:.1%})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
