#!/usr/bin/env python3
"""
Command-line interface for training MIGHTY evasion policy with visualization.

Usage:
    # Headless training with saved plots
    python run_training.py --episodes 1000 --plot-every 20

    # Live visualization (local machine)
    python run_training.py --episodes 500 --live --plot-every 5

    # Quick test (few episodes, plot everything)
    python run_training.py --episodes 20 --plot-every 1 --live

    # Advanced options
    python run_training.py --episodes 1000 --lr 1e-3 --adversary-level 2 --num-steps 50
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_with_viz import train


def main():
    parser = argparse.ArgumentParser(
        description='Train MIGHTY evasion policy with visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Headless training (for remote servers)
  python run_training.py --episodes 1000 --plot-every 20

  # Live visualization (for local machines with display)
  python run_training.py --episodes 500 --live --plot-every 5

  # Quick test run
  python run_training.py --episodes 20 --plot-every 1 --live

  # Advanced: Level 2 adversary with more optimization steps
  python run_training.py --episodes 1000 --adversary-level 2 --num-steps 50

Adversary Levels:
  Level 1: Straight line (constant velocity toward goal) - easier to evade
  Level 2: Proportional navigation (pursues ego actively) - harder to evade
        """
    )

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--adversary-level', type=int, default=1, choices=[1, 2],
                       help='Adversary difficulty: 1=straight line, 2=pursuit (default: 1)')
    parser.add_argument('--num-steps', type=int, default=30,
                       help='Unrolled gradient descent steps for MIGHTY solver (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Visualization parameters
    parser.add_argument('--live', action='store_true',
                       help='Show live matplotlib window (requires display)')
    parser.add_argument('--plot-every', type=int, default=10,
                       help='Visualize every N episodes (default: 10)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization entirely (faster training)')

    # Output parameters
    parser.add_argument('--save-dir', type=str, default='training_results',
                       help='Directory to save results (default: training_results)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 70)
    print("MIGHTY EVASION POLICY TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Training episodes:  {args.episodes}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Adversary level:    {args.adversary_level} " +
          f"({'straight line' if args.adversary_level == 1 else 'proportional pursuit'})")
    print(f"  Solver steps:       {args.num_steps}")
    print(f"  Random seed:        {args.seed}")
    print(f"  Visualization:      {'disabled' if args.no_viz else 'enabled'}")
    if not args.no_viz:
        print(f"  - Live display:     {args.live}")
        print(f"  - Plot every:       {args.plot_every} episodes")
    print(f"  Save directory:     {args.save_dir}")
    print()

    # Warn about live visualization
    if args.live:
        print("⚠️  Live visualization enabled - requires X display or local machine")
        print("   If running on remote server, use --no-viz or remove --live flag")
        print()

    # Confirm for long runs
    if args.episodes > 100:
        print(f"⏱️  Training for {args.episodes} episodes may take a while...")
        print(f"   Estimated time: ~{args.episodes * 2 / 60:.1f} minutes")
        print()

    try:
        # Run training
        policy_params, visualizer = train(
            num_episodes=args.episodes,
            learning_rate=args.lr,
            adversary_level=args.adversary_level,
            save_dir=args.save_dir,
            visualize=not args.no_viz,
            live_viz=args.live,
            plot_every=args.plot_every,
            num_steps=args.num_steps,
            seed=args.seed,
        )

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        # Print summary
        if visualizer is not None:
            history = visualizer.history
            if len(history['episodes']) > 0:
                print("\nFinal Statistics:")
                print(f"  Initial loss:       {history['losses'][0]:7.3f}")
                print(f"  Final loss:         {history['losses'][-1]:7.3f}")
                print(f"  Best loss:          {min(history['losses']):7.3f}")

                print(f"\n  Initial min dist:   {history['min_distances'][0]:5.2f}m")
                print(f"  Final min dist:     {history['min_distances'][-1]:5.2f}m")
                print(f"  Best min dist:      {max(history['min_distances']):5.2f}m")

                print(f"\n  Initial goal dist:  {history['goal_distances'][0]:5.2f}m")
                print(f"  Final goal dist:    {history['goal_distances'][-1]:5.2f}m")
                print(f"  Best goal dist:     {min(history['goal_distances']):5.2f}m")

                conv_rate = sum(history['converged']) / len(history['converged']) * 100
                print(f"\n  Convergence rate:   {conv_rate:5.1f}%")

        print(f"\n📁 Results saved to: {args.save_dir}/")
        if not args.no_viz:
            print(f"📊 Visualizations:   {args.save_dir}/visualizations/")
            print(f"📈 Training curves:  {args.save_dir}/visualizations/training_curves.png")
        print(f"💾 Final model:      {args.save_dir}/final_policy.pkl")

        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print(f"Partial results saved to: {args.save_dir}/")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ ERROR: Training failed")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
