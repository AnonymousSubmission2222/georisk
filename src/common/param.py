import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser
from src.georisk.paths import PROJECT_ROOT, ASSETS_ROOT


@dataclass
class EvalArguments:
    project_prefix: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[2]))
    run_type: str = "eval"
    name: str = "GeoRisk"
    maxWaypoints: int = 200
    batchSize: int = 2
    simulator_tool_port: int = 25000
    DDP_MASTER_PORT: int = 20001
    eval_save_path: Optional[str] = None
    gpu_id: int = 1
    sim_gpu_ids: str = "1,1"
    always_help: bool = True
    use_gt: bool = True
    dataset_path: Optional[str] = None
    eval_json_path: Optional[str] = None
    object_name_json_path: Optional[str] = None
    map_spawn_area_json_path: Optional[str] = None


@dataclass
class ModelArguments:
    model_path: str = str(PROJECT_ROOT / "work_dirs/qwen3vl-uav-2b-lora/checkpoint-7350")
    model_base: str = str(ASSETS_ROOT / "Qwen3-VL-2B-Instruct")
    groundingdino_config: Optional[str] = None
    groundingdino_model_path: Optional[str] = None


@dataclass
class DataArguments:
    view_mode: str = "dual"


parser = HfArgumentParser((EvalArguments, ModelArguments, DataArguments))
args, model_args, data_args = parser.parse_args_into_dataclasses()
if args.run_type != "eval" or data_args.view_mode != "dual":
    raise ValueError("This entry supports GeoRisk dual-view evaluation only")
args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = str(Path(args.project_prefix) / "logs" / f"eval_{args.make_dir_time}.log")
args.machines_info = [{"MACHINE_IP": "127.0.0.1", "SOCKET_PORT": args.simulator_tool_port,
                       "MAX_SCENE_NUM": 16, "open_scenes": []}]
