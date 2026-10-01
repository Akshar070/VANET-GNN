"""Project entry point."""

import sys

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "inspect"

    if command == "parse":
        from src.data.parser import parse_mobility_tcl
        parse_mobility_tcl()
    elif command == "inspect":
        print("Inspection is now integrated into the preprocess stage.")
    elif command == "preprocess":
        from src.data.preprocess import preprocess_mobility_data
        preprocess_mobility_data()
    elif command == "graph":
        from src.graph.pipeline import run_graph_pipeline
        run_graph_pipeline()
    elif command == "kmeans":
        from src.models.kmeans_pipeline import run_kmeans_pipeline
        run_kmeans_pipeline()
    elif command == "spectral":
        from src.models.spectral_pipeline import run_spectral_pipeline
        run_spectral_pipeline()
    elif command == "gae":
        mode = "full"
        if "--test" in sys.argv:
            mode = "test"
        elif "--benchmark" in sys.argv:
            mode = "benchmark"
        from src.models.gae_pipeline import run_gae_pipeline
        run_gae_pipeline(mode=mode)
    elif command == "stable":
        mode = "full"
        if "--test" in sys.argv:
            mode = "test"
        from src.models.stable_pipeline import run_stable_pipeline
        run_stable_pipeline(mode=mode)
    else:
        print(f"Stage selected: {command}")
        print("Implement/run the corresponding module next.")

if __name__ == "__main__":
    main()
