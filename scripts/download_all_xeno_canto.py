import argparse
import json

import pandas as pd
import xenocanto
from joblib import delayed
from tqdm import tqdm

from code_base.utils import ProgressParallel, write_json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species_path",
        type=str,
        default="master_ioc_list_v12.1.xlsx",
        help="Path to file with species",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Slicing indice for species array",
    )
    parser.add_argument(
        "--revert",
        default=False,
        action="store_true",
        help="Wether to revert species array",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=None,
        help="Number of threads",
    )
    parser.add_argument(
        "--inp_path",
        type=str,
        default="dataset",
        help="Path to dataset",
    )
    args = parser.parse_args()
    if args.species_path.endswith(".xlsx"):
        species_info = pd.read_excel(args.species_path)
        all_xeno_canto_species = set(species_info.iloc[:, 9])
        all_xeno_canto_species = [
            el for el in all_xeno_canto_species if isinstance(el, str)
        ]
        write_json("all_xeno_canto_species.json", all_xeno_canto_species)
    elif args.species_path.endswith(".json"):
        all_xeno_canto_species = json.load(open(args.species_path))
    elif args.species_path.endswith(".csv"):
        all_xeno_canto_species = pd.read_csv(args.species_path)[
            "ioc_12_2"
        ].to_list()
        write_json("all_xeno_canto_species.json", all_xeno_canto_species)
    else:
        raise ValueError("Species path must be either .xlsx or .json or .csv")

    if args.start_idx is not None:
        print(f"Slicing from {args.start_idx} idx")
        all_xeno_canto_species = all_xeno_canto_species[args.start_idx :]

    if args.revert:
        print("Reverting array")
        all_xeno_canto_species = list(reversed(all_xeno_canto_species))

    if args.n_threads is None:
        excepted_birds = []
        for selected_bird in tqdm(all_xeno_canto_species):
            try:
                if isinstance(selected_bird, str):
                    xenocanto.download([selected_bird], inp_path=args.inp_path)
                else:
                    try:
                        xenocanto.download(
                            [selected_bird[0]], inp_path=args.inp_path
                        )
                    except:
                        xenocanto.download(
                            [selected_bird[1]], inp_path=args.inp_path
                        )
            except Exception as e:
                print(
                    f"Failed process bird {selected_bird} with exception {e}"
                )
                excepted_birds.append(selected_bird)
    else:

        def load_one_specie(specie_name):
            try:
                if isinstance(specie_name, str):
                    xenocanto.download([specie_name], inp_path=args.inp_path)
                else:
                    try:
                        xenocanto.download(
                            [specie_name[0]], inp_path=args.inp_path
                        )
                    except:
                        xenocanto.download(
                            [specie_name[1]], inp_path=args.inp_path
                        )
                return None
            except Exception as e:
                print(f"Failed process bird {specie_name} with exception {e}")
                return specie_name

        excepted_birds = ProgressParallel(
            n_jobs=args.n_threads,
            total=len(all_xeno_canto_species),
            backend="threading",
        )(
            delayed(load_one_specie)(specie)
            for specie in all_xeno_canto_species
        )
        excepted_birds = [el for el in excepted_birds if el is not None]

    json.dump(excepted_birds, open("excepted_birds.json", "w"))
