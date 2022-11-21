from dataclasses import dataclass, field


@dataclass
class DatasetsArguments:
    dataset_csv_path: str = field(default=None)
