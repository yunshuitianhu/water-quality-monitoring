import os


def resolve_api_key(env_var: str, tool_param: str | None = None) -> str | None:
    """工具参数覆盖优先，其次读取环境变量。"""
    if tool_param:
        return tool_param
    return os.environ.get(env_var)
