#!/usr/bin/env python3
"""
Profile the BPE tokenization training process using cProfile.
"""
import cProfile
import pstats
import sys
import os

from bpe_tokenization import train_bpe_from_config, OPENWEBTEXT_VALID_CONFIG

def profile_bpe_training():
    """Profile BPE training with the sample config."""
    print("Starting BPE training profiling...")
    vocab, merges = train_bpe_from_config(OPENWEBTEXT_VALID_CONFIG)
    print(f"Training completed. Vocab size: {len(vocab)}, Merges: {len(merges)}")
    return vocab, merges

if __name__ == '__main__':
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run profiling
    print("Running cProfile on BPE training...")
    profiler.enable()
    result = profile_bpe_training()
    profiler.disable()
    
    # Save detailed stats to output directory
    output_dir = "/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/cs336_basics/output"
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, f"{OPENWEBTEXT_VALID_CONFIG.output_name}_profile_stats.prof")
    profiler.dump_stats(stats_file)
    print(f"Detailed profiling stats saved to: {stats_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PROFILING RESULTS SUMMARY")
    print("="*80)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("\nTop 20 functions by cumulative time:")
    stats.print_stats(20)
    
    print("\nTop 20 functions by total time (self time):")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    print("\nFunctions with highest per-call time:")
    stats.sort_stats('pcalls')
    stats.print_stats(10)