"""``python -m mayaku.cli``: the same as the ``mayaku`` console script."""

from mayaku.cli.main import main

# guarded: multi-GPU training spawns processes that re-import the main module
if __name__ == "__main__":
    main()
