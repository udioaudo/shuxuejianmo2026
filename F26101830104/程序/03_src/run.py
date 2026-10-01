"""Project entry point. Run from the F working directory."""

import argparse
import json
from pathlib import Path
from fmodel.audit import write_audit
from fmodel.costs import critical_context_length


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["audit","formula-check"])
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    if args.command=="audit":
        target,result=write_audit(args.project)
        print(json.dumps({"report":str(target),"registered_available_count":result['registered_available_count'],
                          "unmapped_file_count":len(result['unmapped_files']),
                          "status":"awaiting_data_or_semantic_validation"},ensure_ascii=False))
    else:
        print(json.dumps({"context_critical_tokens":critical_context_length(),
                          "basis":"problem-defined cost ratio eta*Lctx/6",
                          "scope":"analytic_formula_only_not_empirical_result"},ensure_ascii=False))


if __name__=="__main__":
    main()
