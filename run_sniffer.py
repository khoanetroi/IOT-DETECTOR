"""
run_sniffer.py
Real-time IoT Device Identification from Network Traffic.

This script:
  1. Captures live network packets using Scapy
  2. Extracts flow-level features (packet size, inter-arrival time, etc.)
  3. Feeds them into the trained Transformer model
  4. Prints real-time device identification results

Requirements:
    pip install scapy
    (On Windows: also install Npcap from https://npcap.com)

Usage:
    python run_sniffer.py                           # Auto-detect interface
    python run_sniffer.py --iface "Wi-Fi"           # Specify interface
    python run_sniffer.py --target 192.168.1.50     # Monitor specific IP
"""

import os
import sys
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scapy.all import sniff, IP, TCP, UDP, conf
except ImportError:
    print("[ERROR] Scapy chua duoc cai dat!")
    print("  Chay: .\\python-embed\\python.exe -m pip install scapy")
    print("  Va cai Npcap: https://npcap.com")
    sys.exit(1)

from modules.inference import IoTFingerprinter


# ────────────────────────────────────────────────────────────────
#  FLOW TRACKER
# ────────────────────────────────────────────────────────────────
class FlowTracker:
    """
    Tracks network flows per IP address and extracts
    the same features used during training.
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        # Per-IP packet history
        self.flows = defaultdict(list)  # ip -> list of (timestamp, size, direction)

    def add_packet(self, pkt):
        """Process a captured packet."""
        if not pkt.haslayer(IP):
            return None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        pkt_size = len(pkt)
        timestamp = float(pkt.time)

        # Track source device
        self.flows[src_ip].append({
            "time": timestamp,
            "size": pkt_size,
            "direction": "src",
            "proto": ip_layer.proto,
        })

        # Track destination device
        self.flows[dst_ip].append({
            "time": timestamp,
            "size": pkt_size,
            "direction": "dst",
            "proto": ip_layer.proto,
        })

        return src_ip, dst_ip

    def extract_features(self, ip_addr):
        """
        Extract flow-level features for a given IP address,
        mimicking the columns in the UNSW IoT dataset.
        Returns a numpy array of shape [n_windows, window_size, 17].
        """
        packets = self.flows[ip_addr]
        if len(packets) < self.window_size:
            return None

        # Group into pseudo-flows (every window_size packets = 1 flow)
        features_list = []
        
        for i in range(0, len(packets) - self.window_size + 1, self.window_size):
            window = packets[i : i + self.window_size]
            
            # Calculate features for each "row" in the window
            window_features = []
            for j, pkt_info in enumerate(window):
                # Compute inter-arrival time
                if j > 0:
                    iat = pkt_info["time"] - window[j - 1]["time"]
                else:
                    iat = 0.0

                src_pkts = [p for p in window[:j+1] if p["direction"] == "src"]
                dst_pkts = [p for p in window[:j+1] if p["direction"] == "dst"]
                
                src_sizes = [p["size"] for p in src_pkts] or [0]
                dst_sizes = [p["size"] for p in dst_pkts] or [0]

                # 17 features matching training data
                row = [
                    len(src_pkts),                          # srcNumPackets
                    len(dst_pkts),                          # dstNumPackets
                    sum(src_sizes),                         # srcPayloadSize
                    sum(dst_sizes),                         # dstPayloadSize
                    np.mean(src_sizes),                     # srcAvgPayloadSize
                    np.mean(dst_sizes),                     # dstAvgPayloadSize
                    max(src_sizes),                         # srcMaxPayloadSize
                    max(dst_sizes),                         # dstMaxPayloadSize
                    np.std(src_sizes) if len(src_sizes) > 1 else 0,  # srcStdDevPayloadSize
                    np.std(dst_sizes) if len(dst_sizes) > 1 else 0,  # dstStdDevPayloadSize
                    window[-1]["time"] - window[0]["time"], # flowDuration
                    iat,                                    # srcAvgInterarrivalTime
                    iat,                                    # dstAvgInterarrivalTime
                    iat,                                    # avgInterarrivalTime
                    0.0,                                    # srcStdDevInterarrivalTime
                    0.0,                                    # dstStdDevInterarrivalTime
                    0.0,                                    # stdDevInterarrivalTime
                ]
                window_features.append(row)

            features_list.append(window_features)

        if not features_list:
            return None

        return np.array(features_list, dtype=np.float32)

    def get_tracked_ips(self):
        """Return IPs with enough packets for analysis."""
        return [ip for ip, pkts in self.flows.items() if len(pkts) >= self.window_size]


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time IoT Device Sniffer")
    parser.add_argument("--iface", type=str, default=None, help="Network interface (e.g. 'Wi-Fi')")
    parser.add_argument("--target", type=str, default=None, help="Target IP to monitor")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--checkpoint", type=str, default="experiments/finetune_best.pth")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  IoT Device Sniffer - Real-time Identification")
    print(f"{'='*60}\n")

    # Load AI model
    print("[1] Loading AI model...")
    fingerprinter = IoTFingerprinter(args.checkpoint)

    # Setup sniffer
    tracker = FlowTracker(window_size=10)
    
    bpf_filter = "ip"
    if args.target:
        bpf_filter = f"ip host {args.target}"
        print(f"\n[2] Monitoring target: {args.target}")
    else:
        print(f"\n[2] Monitoring all IP traffic")

    print(f"    Interface: {args.iface or 'auto'}")
    print(f"    Duration: {args.duration}s")
    print(f"    Filter: {bpf_filter}")

    # Capture packets
    print(f"\n[3] Capturing packets for {args.duration} seconds...")
    print(f"    (Make sure ESP devices are connected to the same WiFi!)\n")

    packets = sniff(
        iface=args.iface,
        filter=bpf_filter,
        timeout=args.duration,
        store=True,
    )

    print(f"\n    Captured {len(packets)} packets total.")

    # Process packets
    print("\n[4] Processing flows...")
    for pkt in packets:
        tracker.add_packet(pkt)

    tracked_ips = tracker.get_tracked_ips()
    print(f"    Found {len(tracked_ips)} active IP addresses.\n")

    if not tracked_ips:
        print("[!] No devices with enough traffic detected.")
        print("    Make sure ESP devices are actively sending data.")
        return

    # Identify each device
    print(f"{'='*60}")
    print(f"  IDENTIFICATION RESULTS")
    print(f"{'='*60}\n")

    for ip in tracked_ips:
        windows = tracker.extract_features(ip)
        if windows is None:
            continue

        try:
            results = fingerprinter.predict(windows)

            # Majority vote
            from collections import Counter
            votes = Counter(r["device"] for r in results)
            top_device, top_count = votes.most_common(1)[0]
            confidence = np.mean([r["confidence"] for r in results if r["device"] == top_device])
            total = len(results)

            # Status emoji
            if confidence > 0.7:
                status = "HIGH CONFIDENCE"
            elif confidence > 0.4:
                status = "MEDIUM"
            else:
                status = "UNKNOWN DEVICE (?)"

            print(f"  IP: {ip}")
            print(f"    Predicted Device : {top_device}")
            print(f"    Confidence       : {confidence:.1%}")
            print(f"    Windows analyzed : {total}")
            print(f"    Status           : {status}")
            print(f"    All votes        : {dict(votes)}")
            print()

        except Exception as e:
            print(f"  IP: {ip} -> Error: {e}\n")

    print(f"{'='*60}")
    print(f"  Scan complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
