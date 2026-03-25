"""将本地模型目录推送到 ModelScope（需要有效 Token）。"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="导出到 ModelScope")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True, help="例如: your-name/tiny-r1")
    parser.add_argument("--private", action="store_true", help="是否私有仓库")
    args = parser.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN", "")
    if not token:
        raise ValueError("请先设置环境变量 MODELSCOPE_API_TOKEN")

    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.repository import Repository
    except ImportError as exc:
        raise ImportError("未安装 modelscope，请先 pip install modelscope") from exc

    api = HubApi()
    api.login(token)
    visibility = "private" if args.private else "public"
    api.create_model(args.repo_name, visibility=visibility, exist_ok=True)

    local_repo = os.path.join("outputs", "modelscope_repo")
    os.makedirs(local_repo, exist_ok=True)
    repo = Repository(local_repo, clone_from=args.repo_name)
    repo.push(args.model_dir, commit_message="Upload tiny-r1 model artifacts")
    print(f"[MODELSCOPE] 已推送到: {args.repo_name}")


if __name__ == "__main__":
    main()
