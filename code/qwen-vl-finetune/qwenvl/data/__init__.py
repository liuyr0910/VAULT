"""This release uses explicit --data_path instead of private dataset registries."""
def data_list(dataset_names):
    raise ValueError("Use --data_path with the prepared VAULT dataset JSON.")
