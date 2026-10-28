import streamlit as st

from bocha_search import BochaSearchClient
from config import Config
from llm_client import LLMClient
from memory_extractor import MemoryExtractor
from memory_store import MemoryStore
from persona import PERSONA_NAME, build_system_prompt
from relationship_state import RelationshipState

st.set_page_config(page_title=f"{PERSONA_NAME} · AI 伴侣", page_icon="💬", layout="centered")

_SHOULD_SEARCH_SYSTEM = (
    "你判断用户这句话是否需要联网搜索实时信息（如天气、新闻、日期、股价、事实查询等）。"
    "只需回复一个字：是 或 否。"
)


@st.cache_resource
def get_components():
    Config.validate()
    llm = LLMClient()
    store = MemoryStore()
    extractor = MemoryExtractor(llm)
    bocha = BochaSearchClient()
    rel_state = RelationshipState(llm)
    return llm, store, extractor, bocha, rel_state


def main():
    try:
        llm, store, extractor, bocha, rel_state = get_components()
    except ValueError as e:
        st.error(f"配置错误：{e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── 侧边栏：记忆库管理 ──────────────────────────────────
    with st.sidebar:
        st.title(f"💖 {PERSONA_NAME}")
        st.caption("基于 RAG 记忆的 AI 伴侣 · P1 MVP")
        if not store.available:
            st.warning("⚠️ 记忆功能不可用：本地 embedding 加载失败")
        use_web = st.toggle("🌐 联网模式", value=False, help="打开后，小满会在需要时联网搜索实时信息（天气、新闻、日期等）")
        if use_web and not Config.BOCHA_API_KEY:
            st.warning("联网需配置 BOCHA_API_KEY，请到 open.bocha.cn 注册后填入 .env")
        st.divider()
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("🗑️ 清空记忆库", use_container_width=True):
            store.clear()
            st.success("记忆库已清空")
            st.rerun()
        st.divider()
        st.subheader("🧠 当前记忆库")
        mems = store.get_all_memories()
        if mems:
            for m in mems:
                st.markdown(f"- {m['content']}")
        else:
            st.caption("还没有记忆，聊几句就开始记住你啦")

    # ── 主界面 ─────────────────────────────────────────────
    st.title(f"💬 和 {PERSONA_NAME} 聊天")

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "💖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input("说点什么吧…")
    if not user_input:
        return

    # 1. 展示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2. 检索相关记忆
    memories = store.search(user_input, Config.TOP_K_MEMORIES)
    memories_text = "\n".join(f"- {m['content']}" for m in memories)

    # 3. 组装上下文：人设 + 关系状态 + 记忆
    system_prompt = build_system_prompt(memories_text, rel_state.get())

    # 3.5 联网搜索（开关打开且已配置 key）
    if use_web and Config.BOCHA_API_KEY:
        need_search = llm.invoke(_SHOULD_SEARCH_SYSTEM, user_input)
        if "是" in need_search:
            results = bocha.search(user_input, count=5)
            if results:
                search_text = "\n".join(f"- {r['title']}：{r['snippet']}" for r in results)
                system_prompt += (
                    f"\n\n【联网搜索结果】\n{search_text}\n\n"
                    "请结合以上搜索结果回答用户的问题，但保持你的角色和语气。"
                )

    recent = st.session_state.messages[-Config.RECENT_TURNS * 2:]

    # 4. 流式生成回复（统一用 DeepSeek）
    try:
        with st.chat_message("assistant", avatar="💖"):
            reply = st.write_stream(llm.stream_chat(system_prompt, recent))
    except Exception:
        reply = "（抱歉，服务暂时不可用，请稍后再试）"
        with st.chat_message("assistant", avatar="💖"):
            st.markdown(reply)

    # 5. 保存回复
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 6. 提取本轮记忆并写入（记忆服务不可用时跳过）
    if store.available:
        new_mems = extractor.extract(user_input, reply)
        for mem in new_mems:
            store.add_memory(mem)

    # 7. 更新关系状态摘要
    rel_state.update(user_input, reply)


if __name__ == "__main__":
    main()
