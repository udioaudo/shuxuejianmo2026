"""Inventory and explicit registration audit, without guessing dataset identities."""

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_project(project):
    project = Path(project).resolve()
    raw = (project/"02_data"/"raw").resolve()
    config = json.loads((project/"configs"/"datasets.json").read_text(encoding="utf-8-sig"))
    items, available, bound = [], set(), set()
    for identifier, record in config["datasets"].items():
        files = record.get("paths", [])
        details = []
        for value in files:
            p = (raw/value).resolve()
            if not p.is_relative_to(raw):
                raise ValueError(f"dataset path outside raw directory: {identifier}")
            if not p.is_file():
                details.append({"path":value,"status":"missing_file"})
            else:
                bound.add(p)
                details.append({"path":value,"status":"present_not_semantically_validated",
                                "bytes":p.stat().st_size,"sha256":sha256(p)})
        directory = record.get("directory")
        if directory:
            d = (raw/directory).resolve()
            if not d.is_relative_to(raw):
                raise ValueError(f"dataset directory outside raw directory: {identifier}")
            matched = [p for p in d.rglob(record.get("pattern", "*")) if p.is_file()] if d.is_dir() else []
            bound.update(p.resolve() for p in matched)
            details.append({"directory": directory,
                            "pattern": record.get("pattern", "*"),
                            "matching_files": len(matched),
                            "minimum_files": record.get("minimum_files", 1),
                            "status": "present_not_semantically_validated" if
                            len(matched) >= record.get("minimum_files", 1) else "incomplete"})
        status = "unregistered" if not files and not directory else (
            "files_available_not_validated" if all(
                f.get("status") in ("present_not_semantically_validated",) for f in details)
            else "incomplete")
        if status == "files_available_not_validated":
            available.add(identifier)
        items.append({"id":identifier,"status":status,"files":details})
    requirements = {}
    for name, requirement in config["requirements"].items():
        missing = [x for x in requirement.get("all_of",[]) if x not in available]
        alternatives = []
        for choices in requirement.get("choice_groups",[]):
            satisfied = any(set(option).issubset(available) for option in choices)
            alternatives.append({"choices":choices,"files_satisfied":satisfied})
        requirements[name] = {"missing_registered_inputs":missing,"alternatives":alternatives,
                              "input_files_available":not missing and all(c["files_satisfied"] for c in alternatives),
                              "semantic_validation":"pending",
                              "manual_requirements":requirement.get("manual_requirements",[])}
    candidates=[]
    if raw.exists():
        for p in sorted(raw.rglob("*")):
            if p.is_file() and p.name.lower() not in ("readme.md",".gitkeep") and p.resolve() not in bound:
                candidates.append({"path":str(p.relative_to(raw)),"bytes":p.stat().st_size,
                                   "status":"unmapped_do_not_infer_dataset_id"})
    return {"created_utc":datetime.now(timezone.utc).isoformat(),
            "project":str(project),"raw_directory":str(raw),
            "registered_available_count":len(available),"datasets":items,
            "unmapped_files":candidates,"requirements":requirements,
            "competition_results_ready":False,
            "note":"File presence is not schema, provenance or validation approval."}


def write_audit(project):
    project=Path(project).resolve()
    result=audit_project(project)
    target=project/"08_logs"/"data_audit.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# F题数据登记检查", "", f"已登记且文件存在：{result['registered_available_count']} 项。",
           f"未映射候选文件：{len(result['unmapped_files'])} 个。", "",
           "本检查只核验文件路径与校验值，不把文件存在判作数据有效或赛题完成。", ""]
    for q,row in result['requirements'].items():
        lines += [f"## {q}","", "缺少登记或文件："+", ".join(row['missing_registered_inputs'])]
        for alt in row['alternatives']:
            if not alt['files_satisfied']:
                lines.append("待满足择一组："+" 或 ".join(" + ".join(x) for x in alt['choices']))
        lines += ["还需人工核对："+"；".join(row['manual_requirements']), ""]
    (project/"00_admin"/"当前数据缺件清单.md").write_text("\n".join(lines),encoding="utf-8")
    return target,result
