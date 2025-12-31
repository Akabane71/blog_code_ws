from pathlib import Path

from src.indexer import CACHE_DIR, DATA_DIR, IndexStore
from src.searcher import SearchEngine


def print_menu() -> None:
    print(
        "\n请选择操作:\n"
        "1: 建立/增量更新倒排索引\n"
        "2: 查询文章 (仅返回出处)\n"
        "3: 全量重建倒排索引\n"
        "exit: 退出\n"
    )


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def handle_build(store: IndexStore, full_rebuild: bool = False) -> None:
    stats = store.build_index(full_rebuild=full_rebuild)
    print(
        f"索引完成 | 新增: {stats['added']} 更新: {stats['updated']} "
        f"删除: {stats['removed']} 文档数: {stats['total_docs']} 词汇数: {stats['vocab_size']}"
    )


def handle_search(searcher: SearchEngine) -> None:
    query = input("请输入查询内容: ").strip()
    if not query:
        print("查询内容不能为空。")
        return
    results = searcher.search(query=query, limit=5)
    if not results:
        print("未找到匹配的文章。")
        return
    print("查询结果 (按相关度排序):")
    for idx, (doc_id, score) in enumerate(results, start=1):
        doc_meta = searcher.store.index["docs"].get(doc_id, {})
        print(f"{idx}. {doc_meta.get('path', doc_id)} | score={score:.4f}")


def main() -> None:
    ensure_directories()
    store = IndexStore()
    searcher = SearchEngine(store)

    print("本地模糊文档查询 CLI")
    print_menu()
    while True:
        choice = input("输入操作: ").strip().lower()
        if choice == "1":
            handle_build(store, full_rebuild=False)
        elif choice == "2":
            handle_search(searcher)
        elif choice == "3":
            handle_build(store, full_rebuild=True)
        elif choice == "exit":
            print("已退出。")
            break
        else:
            print("无效输入，请重新选择。")
        print_menu()


if __name__ == "__main__":
    main()
