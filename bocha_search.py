import requests

from config import Config


class BochaSearchClient:
    """博查 Bocha 联网搜索客户端（国内直连，中文搜索质量好）。"""

    ENDPOINT = "https://api.bochaai.com/v1/web-search"

    def search(self, query, count=5):
        if not Config.BOCHA_API_KEY:
            return []
        headers = {
            "Authorization": f"Bearer {Config.BOCHA_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {"query": query, "summary": True, "count": count, "freshness": "noLimit"}
        try:
            resp = requests.post(self.ENDPOINT, headers=headers, json=body, timeout=15)
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as e:
            print(f"[Bocha] 搜索失败：{e}")
            return []

    def _parse(self, data):
        # 兼容两种响应结构：data.webPages.value 或 webPages.value
        pages = (
            data.get("data", {}).get("webPages", {}).get("value", [])
            or data.get("webPages", {}).get("value", [])
        )
        results = []
        for p in pages:
            title = p.get("name") or ""
            url = p.get("url") or ""
            snippet = p.get("summary") or p.get("snippet") or ""
            if not snippet:
                continue
            results.append({"title": title, "url": url, "snippet": snippet})
        return results
