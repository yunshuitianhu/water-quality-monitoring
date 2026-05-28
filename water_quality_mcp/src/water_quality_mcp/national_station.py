import asyncio
import json
import os


class MCPClient:
    """连接到 water-quality-mcp (Node.js) 子进程的客户端。"""

    def __init__(self):
        self.process = None

    async def start(self):
        bin_path = os.environ.get("WATER_QUALITY_MCP_NODE_PATH", "water-quality-mcp")
        self.process = await asyncio.create_subprocess_exec(
            bin_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def stop(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()

    async def call_tool(self, tool_name, arguments=None):
        if not self.process:
            await self.start()
        req = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool_name, "arguments": arguments or {}}, "id": 1}
        self.process.stdin.write((json.dumps(req) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if line:
            res = json.loads(line.decode())
            return res.get("result", res.get("error", "无响应"))
        return "无响应"


async def _call_mcp(tool_name, params=None):
    client = MCPClient()
    await client.start()
    result = await client.call_tool(tool_name, params)
    await client.stop()
    return result


def get_national_station_data(province="", keyword=""):
    """获取国控水质站点数据。"""
    params = {}
    if province:
        params["province"] = province
    if keyword:
        params["search"] = keyword
    try:
        result = asyncio.run(_call_mcp("get_water_quality", params))
        if isinstance(result, dict) and "data" in result:
            return json.dumps(result["data"][:10], ensure_ascii=False)
        return str(result)
    except FileNotFoundError:
        return json.dumps({"error": "water-quality-mcp 未安装或不在 PATH 中。请设置 WATER_QUALITY_MCP_NODE_PATH 环境变量。"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"获取国控站点数据失败: {e}"}, ensure_ascii=False)
